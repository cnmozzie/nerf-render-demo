#include <torch/script.h> // One-stop header.
#include <torch/torch.h>
#include <iostream>
#include <assert.h>

using namespace torch::indexing;


struct NeRF {
    NeRF(int D=8, int W=256, int input_ch=3, int input_ch_views=3, int output_ch=4, int skip=4) {
        this->D = D;
        this->W = W;
        this->input_ch = input_ch;
        this->input_ch_views = input_ch_views;
        this->skip = skip;
    }
    torch::Tensor forward(torch::Tensor x) {
        auto inputs = torch::split(x, {input_ch, input_ch_views}, -1);
        auto input_pts = inputs[0];
        auto input_views = inputs[1];
        
        torch::Tensor h = input_pts.transpose(1,0);
        for(int i=0; i<D; i++) {
            h = torch::matmul(pts_linears_weight[i], h) + pts_linears_bias[i].reshape({-1, 1});
            h = torch::relu(h);
            if (i == skip) {
                h = torch::cat({input_pts.transpose(1,0), h}, 0);
            }
        }
        
        // use viewdirs = true
        auto alpha = torch::matmul(alpha_linear_weight, h) + alpha_linear_bias.reshape({-1, 1});
        auto feature = torch::matmul(feature_linear_weight, h) + feature_linear_bias.reshape({-1, 1});
        h = torch::cat({feature, input_views.transpose(1,0)}, 0);
        // views_linears = [0]
        h = torch::matmul(views_linears_weight, h) + views_linears_bias.reshape({-1, 1});
        h = torch::relu(h);
        auto rgb = torch::matmul(rgb_linear_weight, h) + rgb_linear_bias.reshape({-1, 1});
        auto output = torch::cat({rgb, alpha}, 0);
        
        return output.transpose(1,0);
    }
    void load_weights(c10::List<at::Tensor> weights) {
        // use viewdirs = true
        
        // Load pts_linears
        for (int i=0; i<D; i++) {
            int idx_pts_linears = 2 * i;
            pts_linears_weight.push_back(weights[idx_pts_linears]);
            pts_linears_bias.push_back(weights[idx_pts_linears+1]);
            //std::cout << pts_linears_weight[i].sizes() << std::endl;
        }
        
        // Load feature_linear
        int idx_feature_linear = 2 * D;
        feature_linear_weight = weights[idx_feature_linear];
        feature_linear_bias = weights[idx_feature_linear+1];
        //std::cout << feature_linear_weight.sizes() << std::endl;
        
        // Load views_linears
        int idx_views_linears = 2 * D + 2;
        views_linears_weight = weights[idx_views_linears];
        views_linears_bias = weights[idx_views_linears+1];
        
        // Load rgb_linear
        int idx_rbg_linear = 2 * D + 4;
        rgb_linear_weight = weights[idx_rbg_linear];
        rgb_linear_bias = weights[idx_rbg_linear+1];
        
        // Load alpha_linear
        int idx_alpha_linear = 2 * D + 6;
        alpha_linear_weight = weights[idx_alpha_linear];
        alpha_linear_bias = weights[idx_alpha_linear+1];
    }
    int D;
    int W;
    int input_ch;
    int input_ch_views;
    int skip;
    std::vector<torch::Tensor> pts_linears_weight;
    std::vector<torch::Tensor> pts_linears_bias;
    torch::Tensor feature_linear_weight;
    torch::Tensor feature_linear_bias;
    torch::Tensor views_linears_weight;
    torch::Tensor views_linears_bias;
    torch::Tensor rgb_linear_weight;
    torch::Tensor rgb_linear_bias;
    torch::Tensor alpha_linear_weight;
    torch::Tensor alpha_linear_bias;
};

std::pair<torch::Tensor, int> embed_fn(torch::Tensor inputs, int multires) {
    
    int input_dims = 3;
    int num_freqs = multires;
    int max_freq_log2 = multires - 1;
    int N_freqs = num_freqs;
    int max_freq = max_freq_log2;
    int d = input_dims;
    
    int out_dim = 0;
    // inclue_input = true
    torch::Tensor embedded = inputs;
    out_dim += d;
    
    // log_sampling = true
    auto freq_bands = torch::pow(2, torch::linspace(0., max_freq, N_freqs));
    //std::cout << freq_bands << std::endl;
    
    // 'periodic_fns' : [torch.sin, torch.cos]
    
    for(int i=0; i<N_freqs; i++) {
        embedded = torch::cat({embedded, torch::sin(inputs*freq_bands[i])}, -1);
        embedded = torch::cat({embedded, torch::cos(inputs*freq_bands[i])}, -1);
        out_dim += (2*d);
    }
    
    std::pair<torch::Tensor, int> pair(embedded, out_dim);
    
    return pair;
}

torch::Tensor run_network(torch::Tensor inputs, torch::Tensor viewdirs, int netchunk=1024*64) {
    
    auto inputs_flat = inputs.reshape({-1, inputs.size(-1)});
    
    // log2 of max freq for positional encoding (3D location)
    int multires = 10;
    std::pair<torch::Tensor, int> pair = embed_fn(inputs_flat, multires);
    auto embedded = pair.first;
    int input_ch = pair.second;
    
    // viewdirs is not None
    auto input_dirs = viewdirs.index({Slice(), None}).expand_as(inputs);
    auto input_dirs_flat = input_dirs.reshape({-1, input_dirs.size(-1)});
    // log2 of max freq for positional encoding (2D direction)
    int multires_views = 4;
    pair = embed_fn(input_dirs_flat, multires_views);
    auto embedded_dirs = pair.first;
    int input_ch_views = pair.second;
    embedded = torch::cat({embedded, embedded_dirs}, -1);
    
    // create_nerf
    int netdepth = 8;
    int netwidth = 256;
    int N_importance = 0; // for fast render
    int output_ch = N_importance > 0 ? 5 : 4;
    int skip = 4;
    // use viewdirs = true
    NeRF model(netdepth, netwidth, input_ch, input_ch_views, output_ch, skip);
    
    // Load checkpoints
    torch::jit::script::Module container = torch::jit::load("container.pt");
    c10::List<at::Tensor> weights = container.attr("weights").toTensorList();
    model.load_weights(weights);
    
    //seems ok here
    //std::cout << embedded_dirs[50] << std::endl;
    //std::cout << embedded[50] << std::endl;
    
    torch::Tensor outputs_flat;
    for(int i=0; i<embedded.size(0); i+=netchunk) {
        //std::cout << i << "\t" << i+netchunk << std::endl;
        auto output = model.forward(embedded.index({Slice(i, i+netchunk)}));
        if (outputs_flat.size(0) == 0) {
            outputs_flat = output;
        }
        else {
            outputs_flat = torch::cat({outputs_flat, output}, 0);
        }
    }
    
    //std::cout << inputs << std::endl;
    //std::cout << outputs_flat << std::endl;
    
    torch::Tensor outputs = outputs_flat.reshape({inputs.size(0), inputs.size(1), outputs_flat.size(-1)});
    
    return outputs;
    
}


std::vector<torch::Tensor> raw2outputs(torch::Tensor raw, torch::Tensor z_vals, torch::Tensor rays_d) {
    //std::cout << raw << std::endl;
    //std::cout << torch::relu(raw.index({"...", 3})) << std::endl;
    
    torch::Tensor dists = z_vals.index({"...", Slice(1, None)}) - z_vals.index({"...", Slice(None, -1)});
    dists = torch::cat({dists, torch::tensor({1e10}).expand_as(dists.index({"...", Slice(None, 1)}))}, -1);
    dists = dists * torch::norm(rays_d.index({"...", None, Slice()}), 2, -1);
    
    auto rgb = torch::sigmoid(raw.index({"...", Slice(None,3)}));
    double noise = 0.0;
    // raw_noise_std = 0
    // if raw_noise_std > 0, add some noise ...
    auto alpha = 1.0 - torch::exp(-torch::relu(raw.index({"...", 3})+noise)*dists);
    // cumprod is use to check if alpha_n is blocked by other alpha_i
    torch::Tensor weights = alpha * torch::cumprod(torch::cat({torch::ones({alpha.size(0), 1}), 1.0-alpha+1e-10}, -1), -1).index({Slice(), Slice(None, -1)});
    torch::Tensor rgb_map = torch::sum(weights.index({"...", None}) * rgb, -2);
    torch::Tensor depth_map = torch::sum(weights * z_vals, -1);
    torch::Tensor acc_map = torch::sum(weights, -1);
    torch::Tensor disp_map = 1.0 / torch::max(1e-10 * torch::ones_like(depth_map), depth_map / acc_map); // could be nan
    
    // white_bkgd = false
    // if true, rgb_map = rgb_map + (1.-acc_map[...,None])
    
    //std::cout << rgb_map << std::endl;
    //std::cout << disp_map << std::endl;
    //std::cout << acc_map << std::endl;
    //std::cout << weights << std::endl;
    //std::cout << depth_map << std::endl;
    
    std::vector<torch::Tensor> outputs{rgb_map, disp_map, acc_map, weights, depth_map};
    
    return outputs;
}


std::vector<torch::Tensor> render_rays(torch::Tensor ray_batch, int N_samples=64, float perturb=0.) {
    int N_rays = ray_batch.size(0);
    auto rays_o = ray_batch.index({Slice(), Slice(0, 3)});
    auto rays_d = ray_batch.index({Slice(), Slice(3, 6)});
    
    
    // use viewdirs = true
    auto viewdirs = ray_batch.index({Slice(), Slice(-3, None)});
    auto bounds = ray_batch.index({"...", Slice(6, 8)}).reshape({-1,1,2});
    auto near = bounds.index({"...", 0});
    auto far = bounds.index({"...", 1});
    
    auto t_vals = torch::linspace(0., 1., N_samples);
    
    // lindisp = false
    auto z_vals = near * (1.-t_vals) + far * (t_vals);
    
    // If perturb non-zero, each ray is sampled at stratified random points in time.
    // ...
    
    // [ray_id, sample_id, xyz]
    auto pts = rays_o.index({"...", None, Slice()}) + rays_d.index({"...", None, Slice()}) * z_vals.index({"...", Slice(), None});
    
    // seems ok here
    //std::cout << pts[5] << std::endl;
    
    int netchunk = 1024*64;
    auto raw = run_network(pts, viewdirs, netchunk);
    
    std::vector<torch::Tensor> ret = raw2outputs(raw, z_vals, rays_d);
    
    // N_importance = 0, not need to sample further
    
    return ret;
    
}

std::vector<torch::Tensor> batchify_rays(torch::Tensor rays_flat, int chunk=1024*32) {
    torch::Tensor rgb_map, disp_map, acc_map;
    for(int i=0; i<rays_flat.size(0); i+=chunk) {
        //std::cout << i << "\t" << i+chunk << std::endl;
        auto ret = render_rays(rays_flat.index({Slice(i, i+chunk)}));
        if (rgb_map.size(0) == 0) {
            rgb_map = ret[0];
            disp_map = ret[1];
            acc_map = ret[2];
        }
        else {
            rgb_map = torch::cat({rgb_map, ret[0]}, 0);
            disp_map = torch::cat({disp_map, ret[1]}, 0);
            acc_map = torch::cat({acc_map, ret[2]}, 0);
        }
    }
    //std::cout << rgb_map << std::endl;
    
    std::vector<torch::Tensor> all_ret{rgb_map, disp_map, acc_map};
    return all_ret;
    
    // test only  (10,11)*6
    // auto ret = render_rays(rays_flat.index({Slice(0, 3)}), 6);
}

std::vector<torch::Tensor> get_rays(int H, int W, torch::Tensor K, torch::Tensor c2w) {
    //std::cout << H << ", " << W << std::endl;
    //std::cout << K << std::endl;
    //std::cout << c2w << std::endl;
    
    auto cood = torch::meshgrid({torch::linspace(0, W-1, W), torch::linspace(0, H-1, H)});
    auto i = cood[0].transpose(1,0);
    auto j = cood[1].transpose(1,0);
    
    auto dirs = torch::stack({(i-K.index({0,2}))/K.index({0,0}), -(j-K.index({1,2}))/K.index({1,1}), -torch::ones_like(i)}, -1);
    auto rays_d = torch::empty_like(dirs);
    
    for(int w=0; w<W; w++)
        for(int h=0; h<H; h++) {
            rays_d.index_put_({h, w}, torch::matmul(c2w.index({Slice(None,3), Slice(None,3)}), dirs.index({h, w})));
        }
    
    auto rays_o = c2w.index({Slice(None,3), -1}).expand_as(rays_d);
    
    std::vector<torch::Tensor> rays{rays_d, rays_o};

    return rays;
}

std::vector<torch::Tensor> ndc_rays(int H, int W, double focal, double near, torch::Tensor rays_o, torch::Tensor rays_d) {
    
    // Shift ray origins to near plane
    auto t = -(near + rays_o.index({"...", 2})) / rays_d.index({"...", 2});
    rays_o = rays_o + t.index({"...", None}) * rays_d;
    
    
    // Projection
    auto o0 = -1.0/(W/(2.0*focal)) * rays_o.index({"...", 0}) / rays_o.index({"...", 2});
    auto o1 = -1.0/(H/(2.0*focal)) * rays_o.index({"...", 1}) / rays_o.index({"...", 2});
    auto o2 = 1.0 + 2.0 * near / rays_o.index({"...", 2});
    
    auto d0 = -1.0/(W/(2.0*focal)) * (rays_d.index({"...", 0})/rays_d.index({"...", 2}) - rays_o.index({"...", 0}) / rays_o.index({"...", 2}));
    auto d1 = -1.0/(H/(2.0*focal)) * (rays_d.index({"...", 1})/rays_d.index({"...", 2}) - rays_o.index({"...", 1}) / rays_o.index({"...", 2}));
    auto d2 = -2.0 * near / rays_o.index({"...", 2});
    
    
    rays_o = torch::stack({o0, o1, o2}, -1);
    rays_d = torch::stack({d0, d1, d2}, -1);
    
    //std::cout << rays_o << std::endl;
    //std::cout << rays_d << std::endl;

    std::vector<torch::Tensor> rays{rays_d, rays_o};
    
    return rays;
}


std::vector<torch::Tensor> render(int H, int W, torch::Tensor K, torch::Tensor c2w) {
    // c2w is not None
    std::vector<torch::Tensor> _rays = get_rays(H, W, K, c2w);
    auto rays_d = _rays[0];
    auto rays_o = _rays[1];
    
    // use viewdirs = true
    auto viewdirs = rays_d;
    // s2w_staticcam is None
    viewdirs = viewdirs / torch::norm(viewdirs, 2, -1, true);
    viewdirs = viewdirs.reshape({-1,3});
    
    int h = rays_d.size(0);
    int w = rays_d.size(1);
    
    // no_ndc=False, forward facing scenes
    _rays = ndc_rays(H, W, K[0][0].item<double>(), 1.0, rays_o, rays_d);
    rays_d = _rays[0];
    rays_o = _rays[1];
    
    rays_o = rays_o.reshape({-1,3});
    rays_d = rays_d.reshape({-1,3});
    
    double _near = 0.0;
    double _far = 1.0;
    auto near = _near * torch::ones_like(rays_d.index({"...", Slice(None,1)}));
    auto far = _far * torch::ones_like(rays_d.index({"...", Slice(None,1)}));
    
    auto rays = torch::cat({rays_o, rays_d, near, far}, -1);

    
    // use viewdirs = true
    rays = torch::cat({rays, viewdirs}, -1);
    
    //std::cout << rays << std::endl;
    
    int chunk = 32768;
    auto all_ret = batchify_rays(rays, chunk);
    
    
    auto rgb_map = all_ret[0].reshape({h, w, all_ret[0].size(1)});
    auto disp_map = all_ret[1].reshape({h, w});
    auto acc_map = all_ret[2].reshape({h, w});
    
    std::cout << rgb_map.sizes() << std::endl;
    std::cout << disp_map.sizes() << std::endl;
    std::cout << acc_map.sizes() << std::endl;
    
    
    
    std::vector<torch::Tensor> ret_list{rgb_map, disp_map, acc_map};
    
    return ret_list;
    
}


int main() {
    int H = 756;
    int W = 1008;
    double focal = 815.1316;
    
    torch::Tensor K = torch::tensor({{focal, 0., 0.5*W}, \
                            {0., focal, 0.5*H}, \
                            {0., 0., 1.}});
                            
    torch::Tensor c2w = torch::eye(4).index({Slice(None, 3), Slice(None)});
    
    int down = 4;
    
    // 3*4 for now
    auto test = render(H/down, W/down, K/down, c2w);
    auto rgb_map = test[0].clip(0.0, 1.0);
    auto disp_map = test[1];
    auto acc_map = test[2];
    
    //std::cout << img[0] << std::endl;
    
    // test save
    torch::save({rgb_map, disp_map, acc_map}, "tensors.pt");
    
    
    
    

}

