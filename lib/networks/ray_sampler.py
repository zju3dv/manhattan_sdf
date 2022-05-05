import numpy as np
import torch


def sdf_to_sigma(sdf: torch.Tensor, alpha, beta):
    exp = 0.5 * torch.exp(-torch.abs(sdf) / beta)
    psi = torch.where(sdf >= 0, exp, 1 - exp)
    return alpha * psi


def error_bound(d_vals, sdf, alpha, beta):
    device = sdf.device
    sigma = sdf_to_sigma(sdf, alpha, beta)
    sdf_abs_i = torch.abs(sdf)
    delta_i = d_vals[..., 1:] - d_vals[..., :-1]

    R_t = torch.cat(
        [
            torch.zeros([*sdf.shape[:-1], 1], device=device), 
            torch.cumsum(sigma[..., :-1] * delta_i, dim=-1)
        ], dim=-1)[..., :-1]

    d_i_star = torch.clamp_min(0.5 * (sdf_abs_i[..., :-1] + sdf_abs_i[..., 1:] - delta_i), 0.)
    errors = alpha/(4*beta) * (delta_i**2) * torch.exp(-d_i_star / beta)
    errors_t = torch.cumsum(errors, dim=-1)
    bounds = torch.exp(-R_t) * (torch.exp(errors_t) - 1.)
    bounds[torch.isnan(bounds)] = np.inf
    return bounds


def sample_pdf(bins, weights, N_importance, det=False, eps=1e-5):
    # device = weights.get_device()
    device = weights.device
    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat(
        [torch.zeros_like(cdf[..., :1], device=device), cdf], -1
    )  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0.0, 1.0, steps=N_importance, device=device)
        u = u.expand(list(cdf.shape[:-1]) + [N_importance])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_importance], device=device)
    u = u.contiguous()

    # Invert CDF
    inds = torch.searchsorted(cdf.detach(), u, right=False)

    below = torch.clamp_min(inds-1, 0)
    above = torch.clamp_max(inds, cdf.shape[-1]-1)
    # (batch, N_importance, 2) ==> (B, batch, N_importance, 2)
    inds_g = torch.stack([below, above], -1)

    matched_shape = [*inds_g.shape[:-1], cdf.shape[-1]]  # fix prefix shape

    cdf_g = torch.gather(cdf.unsqueeze(-2).expand(matched_shape), -1, inds_g)
    bins_g = torch.gather(bins.unsqueeze(-2).expand(matched_shape), -1, inds_g)  # fix prefix shape

    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom[denom<eps] = 1
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples


def sample_cdf(bins, cdf, N_importance, det=False, eps=1e-5):
    # device = weights.get_device()
    device = bins.device
    cdf = torch.cat(
        [torch.zeros_like(cdf[..., :1], device=device), cdf], -1
    )  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0.0, 1.0, steps=N_importance, device=device)
        u = u.expand(list(cdf.shape[:-1]) + [N_importance])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_importance], device=device)
    u = u.contiguous()

    # Invert CDF
    inds = torch.searchsorted(cdf.detach(), u, right=False)

    below = torch.clamp_min(inds-1, 0)
    above = torch.clamp_max(inds, cdf.shape[-1]-1)
    # (batch, N_importance, 2) ==> (B, batch, N_importance, 2)
    inds_g = torch.stack([below, above], -1)

    matched_shape = [*inds_g.shape[:-1], cdf.shape[-1]]  # fix prefix shape

    cdf_g = torch.gather(cdf.unsqueeze(-2).expand(matched_shape), -1, inds_g)
    bins_g = torch.gather(bins.unsqueeze(-2).expand(matched_shape), -1, inds_g)  # fix prefix shape

    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom[denom<eps] = 1
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples


def fine_sample(sdf_net_fn, init_dvals, rays_o, rays_d, 
                alpha_net, beta_net, far, 
                eps=0.1, max_iter:int=5, max_bisection:int=10, final_N_importance:int=64, N_up:int=128,
                perturb=True):

    with torch.no_grad():
        device = init_dvals.device
        prefix = init_dvals.shape[:-1]
        d_vals = init_dvals
        
        def query_sdf(d_vals_, rays_o_, rays_d_):
            pts = rays_o_[..., None, :] + rays_d_[..., None, :] * d_vals_[..., :, None]
            return sdf_net_fn(pts)

        def opacity_invert_cdf_sample(d_vals_, sdf_, alpha_, beta_, N_importance=final_N_importance, det=not perturb):
            sigma = sdf_to_sigma(sdf_, alpha_, beta_)
            delta_i = d_vals_[..., 1:] - d_vals_[..., :-1]
            R_t = torch.cat(
                [
                    torch.zeros([*sdf_.shape[:-1], 1], device=device), 
                    torch.cumsum(sigma[..., :-1] * delta_i, dim=-1)
                ], dim=-1)[..., :-1]
                
            opacity_approx = 1 - torch.exp(-R_t)
            fine_dvals = sample_cdf(d_vals_, opacity_approx, N_importance, det=det)
            return fine_dvals

        final_fine_dvals = torch.zeros([*prefix, final_N_importance]).to(device)
        final_iter_usage = torch.zeros([*prefix]).to(device)

        if not isinstance(far, torch.Tensor):
            far = far * torch.ones([*prefix, 1], device=device)
        beta = torch.sqrt((far**2) / (4 * (init_dvals.shape[-1]-1) * np.log(1+eps)))
        alpha = 1./beta
        
        sdf = query_sdf(d_vals, rays_o, rays_d)
        net_bounds_max = error_bound(d_vals, sdf, alpha_net, beta_net).max(dim=-1).values
        mask = net_bounds_max > eps
        
        bounds = error_bound(d_vals, sdf, alpha, beta)
        bounds_masked = bounds[mask]
        
        final_converge_flag = torch.zeros([*prefix], device=device, dtype=torch.bool)
        
        if (~mask).sum() > 0:
            final_fine_dvals[~mask] = opacity_invert_cdf_sample(d_vals[~mask], sdf[~mask], alpha_net, beta_net)
            final_iter_usage[~mask] = 0
        final_converge_flag[~mask] = True
        
        cur_N = init_dvals.shape[-1]
        it_algo = 0
        
        while it_algo < max_iter:
            it_algo += 1
            if mask.sum() > 0:
                upsampled_d_vals_masked = sample_pdf(d_vals[mask], bounds_masked, N_up+2, det=True)[..., 1:-1]
                
                d_vals = torch.cat([d_vals, torch.zeros([*prefix, N_up]).to(device)], dim=-1)
                sdf = torch.cat([sdf, torch.zeros([*prefix, N_up]).to(device)], dim=-1)

                d_vals_masked = d_vals[mask]
                sdf_masked = sdf[mask]
                d_vals_masked[..., cur_N:cur_N+N_up] = upsampled_d_vals_masked
                d_vals_masked, sort_indices_masked = torch.sort(d_vals_masked, dim=-1)
                sdf_masked[..., cur_N:cur_N+N_up] = query_sdf(upsampled_d_vals_masked, rays_o[mask], rays_d[mask])
                sdf_masked = torch.gather(sdf_masked, dim=-1, index=sort_indices_masked)
                d_vals[mask] = d_vals_masked
                sdf[mask] = sdf_masked
                
                cur_N += N_up

                net_bounds_max[mask] = error_bound(d_vals[mask], sdf[mask], alpha_net, beta_net).max(dim=-1).values

                sub_mask_of_mask = net_bounds_max[mask] > eps

                converged_mask = mask.clone()
                converged_mask[mask] = ~sub_mask_of_mask

                if converged_mask.sum() > 0:
                    final_converge_flag[converged_mask] = True
                    final_fine_dvals[converged_mask] = opacity_invert_cdf_sample(d_vals[converged_mask], sdf[converged_mask], alpha_net, beta_net)
                    final_iter_usage[converged_mask] = it_algo

                if (sub_mask_of_mask).sum() > 0:

                    new_mask = mask.clone()
                    new_mask[mask] = sub_mask_of_mask

                    beta_right = beta[new_mask]
                    beta_left = beta_net * torch.ones_like(beta_right, device=device)
                    d_vals_tmp = d_vals[new_mask]
                    sdf_tmp = sdf[new_mask]

                    for _ in range(max_bisection):
                        beta_tmp = 0.5 * (beta_left + beta_right)
                        alpha_tmp = 1./beta_tmp

                        bounds_tmp_max = error_bound(d_vals_tmp, sdf_tmp, alpha_tmp, beta_tmp).max(dim=-1).values
                        beta_right[bounds_tmp_max <= eps] = beta_tmp[bounds_tmp_max <= eps]
                        beta_left[bounds_tmp_max > eps] = beta_tmp[bounds_tmp_max > eps]
                    beta[new_mask] = beta_right
                    alpha[new_mask] = 1./beta[new_mask]

                    bounds_masked = error_bound(d_vals_tmp, sdf_tmp, alpha[new_mask], beta[new_mask])
                    bounds_masked = torch.clamp(bounds_masked, 0, 1e5)
                    
                    mask = new_mask
                else:
                    break
            else:
                break

        if (~final_converge_flag).sum() > 0:
            beta_plus = beta[~final_converge_flag]
            alpha_plus = 1./beta_plus
            final_fine_dvals[~final_converge_flag] = opacity_invert_cdf_sample(d_vals[~final_converge_flag], sdf[~final_converge_flag], alpha_plus, beta_plus)
            final_iter_usage[~final_converge_flag] = -1
        beta[final_converge_flag] = beta_net
        return final_fine_dvals, beta, final_iter_usage
