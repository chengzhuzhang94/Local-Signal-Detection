function [records] = CZ_bootstrap_customized(seed, lower,upper,nlam,TRI, mat_Z, Z, n, nc, d, nv, v1, v2, v3, e1, e2, e3, ie1, m, kLoopTime, option1, option2, a, a_tune)
% This function seems to fit all varying coefficient functions together, not separately
nt = size(TRI, 1);
b_hat = (transpose(mat_Z) * mat_Z + 1 / (log(n)*nt) * eye(m*nc)) \ transpose(mat_Z) * Z;
threshold = 10 ^ (-3);
lam_vec = linspace(lower, upper, nlam);
bic = zeros(nlam, 1); converged_or_not = zeros(1, nlam);
for q = 1:nlam
    [p_b_hat, dist_logical] = update_p_b_hat_2(mat_Z, Z, b_hat, threshold, lam_vec(q), a, m, nc, d, nv, v1, v2, v3, e1, e2, e3, ie1, TRI, 300, n,2,1);
    converged_or_not(q) = dist_logical; %p_b_hat(abs(p_b_hat)<0.00001) = 0;
    bic(q) = log(mean((Z - mat_Z * p_b_hat).^2)) + log(n) * sum(p_b_hat ~=0) / n;
end

[temp_min, temp_index] = min(bic); fitted_lambda=lam_vec(temp_index) ;
if option2 ~= 3
    [p_b_hat, dist_logical] = update_p_b_hat_2(mat_Z, Z, b_hat, threshold, lam_vec(temp_index), a, m, nc, d, nv, v1, v2, v3, e1, e2, e3, ie1, TRI, 300, n,2,1);
else
    [p_b_hat, dist_logical] = update_p_b_hat_2(mat_Z, Z, b_hat, threshold, lam_vec(temp_index), a-a_tune, m, nc, d, nv, v1, v2, v3, e1, e2, e3, ie1, TRI, 300, n,2,1);
end


if option1 == 1
    Res = Z - mat_Z*p_b_hat; Z_bootbase = mat_Z*p_b_hat;
elseif option1 == 2
    Res = Z - mat_Z*b_hat; Z_bootbase = mat_Z*b_hat;
elseif option1 == 0
    Res = Z - mat_Z*p_b_hat; Z_bootbase = mat_Z*p_b_hat; Res = Res - mean(Res);
end

if seed ~= 0
    rng(seed);
end
records = zeros(kLoopTime, size(TRI,1)*m); lambda_records = zeros(1, kLoopTime); %comparisons = zeros(4, kLoopTime);
for loop = 1:kLoopTime
    if option1 == 1 || option1 == 2
        %Z_i_W = ones(length(Z), 1) .* (-sqrt(5)+1)/2; temp_indices = find(rand(length(Z), 1) < (5-sqrt(5))/10); Z_i_W(temp_indices) = (sqrt(5)+1)/2;
        %temp_Z = Z_bootbase + Res .* Z_i_W; temp_mat_Z = mat_Z;% Implement wild bootstrap based on penalized fitting        
        Z_i_W = ones(length(Z), 1) .* (-sqrt(5)+1)/2;  Z_i_W(rand(length(Z), 1) < (5-sqrt(5))/10) = (sqrt(5)+1)/2;
        temp_Z = Z_bootbase + Res .* Z_i_W; temp_mat_Z = mat_Z;% Implement wild bootstrap based on penalized fitting
    elseif option1 == 3
        temp_order = datasample(1:n, n); temp_Z = Z(temp_order); temp_mat_Z = mat_Z(temp_order, :);
    elseif option1 == 0
        temp_order = datasample(1:n, n); temp_Z = Z_bootbase + Res(temp_order); temp_mat_Z = mat_Z; temp_Res = Res(temp_order);
    end
    
    b_hat = (transpose(temp_mat_Z) * temp_mat_Z + 1 / (log(n)*nt) * eye(m*nc)) \ transpose(temp_mat_Z) * temp_Z; 
    
    if option2 == 1
        threshold = 10 ^ (-3); 
        lam_vec = linspace(lower, upper, nlam);
        bic = zeros(nlam, 1); converged_or_not = zeros(1, nlam);
        for q = 1:nlam
            [p_b_hat, dist_logical] = update_p_b_hat_2(temp_mat_Z, temp_Z, b_hat, threshold, lam_vec(q), a, m, nc, d, nv, v1, v2, v3, e1, e2, e3, ie1, TRI, 300, n,2,1);
            converged_or_not(q) = dist_logical; 
            bic(q) = log(mean((temp_Z - temp_mat_Z * p_b_hat).^2)) + log(n) * sum(p_b_hat ~= 0) / n;
        end        
        [temp_min, temp_index] = min(bic); lambda_records(loop)=lam_vec(temp_index); % Use the optimized lambda value
        [p_b_hat, dist_logical] = update_p_b_hat_2(temp_mat_Z, temp_Z, b_hat, threshold, lam_vec(temp_index), a, m, nc, d, nv, v1, v2, v3, e1, e2, e3, ie1, TRI, 300, n,2,1); %comparisons(1, loop)=lam_vec(temp_index);comparisons(2, loop)=sum(p_b_hat==0);
    elseif option2 == 2
        [p_b_hat, dist_logical] = update_p_b_hat_2(temp_mat_Z, temp_Z, b_hat, threshold, fitted_lambda, a, m, nc, d, nv, v1, v2, v3, e1, e2, e3, ie1, TRI, 300, n,2,1); %comparisons(3, loop)=fitted_lambda;comparisons(4, loop)=sum(p_b_hat==0);
    elseif option2 == 3
        [p_b_hat, dist_logical] = update_p_b_hat_2(temp_mat_Z, temp_Z, b_hat, threshold, fitted_lambda, a-a_tune, m, nc, d, nv, v1, v2, v3, e1, e2, e3, ie1, TRI, 300, n,2,1);
    end
    
    [res1, res2] = p_b_hat2TRI_NO(p_b_hat, TRI, m);
    records(loop, :) = res1;
    
end

end



