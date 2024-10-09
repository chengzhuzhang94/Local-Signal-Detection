function [records, lambda_records] = CZ_bootstrap_logic_nested(seed, TRI, pv, vx, vy, n, mat_Z, Z, b_hat, p_b_hat, nt, nc, nv, d, v1, v2, v3, e1, e2, e3, ie1, m, kLoopTime, bootstrap_opt, lambda_opt, optimized_lambda)
% It's used in the nested loop of the function `CZ_bootstrap_logic`
% Suppose we already have: mat_Z, Z, b_hat, p_b_hat

fixed_b_hat = b_hat; fixed_p_b_hat = p_b_hat; 
% Output: records of estimated null triangles
records = zeros(kLoopTime, nt*m); lambda_records = zeros(kLoopTime, m);
rng(seed);

a = 3.7; threshold = 10 ^ (-3); 

% bootstrap type 1: Vector resampling; All varying coefficient functions are estiamted simultaneously
if bootstrap_opt==1
    for loop=1:kLoopTime
        % Resampling to get bootstrap IDs
        bootstrap_id = datasample(1:n, n);
        temp_mat_Z = mat_Z(bootstrap_id, :); temp_Z = Z(bootstrap_id); 
        [temp_b_hat dev stats] = glmfit(temp_mat_Z, temp_Z, 'binomial', 'link', 'logit', 'constant', 'off');
        
        if lambda_opt == 1
            % This option directly applies the optimized_lambda from original dataset
            lambda_records(loop, :) = optimized_lambda;
            [p_b_hat, dist_logical, probs, first_deri_l, second_deri_l, Dgn, accuracy_record] = update_p_b_hat_logistic_NoC(...
                temp_mat_Z, temp_Z, temp_b_hat, threshold, optimized_lambda, a, m, nc, d, nv, v1, v2, v3, e1, e2, e3, ie1, TRI, 30, n, 1);
            [res1, res2] = p_b_hat2TRI_NO(p_b_hat, TRI, m); records(loop, :) = res1;  
            
        elseif lambda_opt == 0
            % This option would find the optimized lambda
            nlam = 7; a = 3.7;threshold = 10 ^ (-3); lam_vec = linspace(0.6, 0.9, nlam);
            bic = zeros(nlam, 1); converged_or_not = zeros(nlam, 1);
            for q = 1:nlam
                [p_b_hat, dist_logical, probs, first_deri_l, second_deri_l, Dgn, accuracy_record] = update_p_b_hat_logistic_NoC(...
                    temp_mat_Z, temp_Z, temp_b_hat, threshold, lam_vec(q), a, m, nc, d, nv, v1, v2, v3, e1, e2, e3, ie1, TRI, 30, n, 1);
                converged_or_not(q) = dist_logical; preds = cal_probs(temp_mat_Z * p_b_hat);

                preds_valid = setdiff(setdiff(1:1:n, find(temp_Z==1 & preds==1)), find(temp_Z==0 & preds==0)) ;
                bic(q, 1) =  -2*sum(temp_Z(preds_valid).*log(preds(preds_valid))) - 2*sum((1-temp_Z(preds_valid)).*log(1-preds(preds_valid))) + log(length(preds_valid)) * sum(p_b_hat ~= 0);
            end

            [temp_min, temp_index] = min(bic(:, 1)); lambda_records(loop, :) = lam_vec(temp_index); 
            [p_b_hat, dist_logical, probs, first_deri_l, second_deri_l, Dgn, accuracy_record] = update_p_b_hat_logistic_NoC(...
                temp_mat_Z, temp_Z, temp_b_hat, threshold, lam_vec(temp_index), a, m, nc, d, nv, v1, v2, v3, e1, e2, e3, ie1, TRI, 30, n, 1);
            [res1, res2] = p_b_hat2TRI_NO(p_b_hat, TRI, m); records(loop, :) = res1;  
        end
         
    end
end

% bootstrap type 2: Parametric bootstrap confidence interval. Varying coefficient functions are estimated respectively
% In this type, we use p_b_hat to draw bootstrap samples, not b_hat; mat_Z is fixed

if bootstrap_opt==2
    optimized_lambda = zeros(1, m);
    for loop=1:kLoopTime
        for vcf_no=1:m
            % vcf are the initial of "Varying Coefficient Functions". In this loop, we generate & estimate the vcf_no-th
            % varying coefficient function
            
            % Eliminate other VCFs' effects
            temp_mat_Z = mat_Z(:, (vcf_no-1)*nc+1:vcf_no*nc);
            temp_Z = binornd(1, cal_probs(temp_mat_Z * fixed_p_b_hat((vcf_no-1)*nc+1:vcf_no*nc) )); 
            %temp_esti_b_hat = ;
            
            [vcf_b_hat dev stats] = glmfit(temp_mat_Z, temp_Z, 'binomial', 'link', 'logit', 'constant', 'off');
            
            % This version can only handle two types of lambda_opt
            % Either opt would init optimized_lambda since they were 0s. Then if lambda_opt==1, the shorter version is used
            if optimized_lambda(vcf_no) ~= 0 && lambda_opt == 1
                [p_b_hat, dist_logical, probs, first_deri_l, second_deri_l, Dgn, accuracy_record] = update_p_b_hat_logistic_NoC(...
                    temp_mat_Z, temp_Z, vcf_b_hat, threshold, optimized_lambda, a, 1, nc, d, nv, v1, v2, v3, e1, e2, e3, ie1, TRI, 30, n, 1);
                [res1, res2] = p_b_hat2TRI_NO(p_b_hat, TRI, 1); records(loop, (vcf_no-1)*nt+1:vcf_no*nt) = res1;
            else                
                nlam = 16; a = 3.7;threshold = 10 ^ (-3); lam_vec = linspace(0.05, 0.8, nlam);
                bic = zeros(nlam, 1); converged_or_not = zeros(nlam, 1);
                for q = 1:nlam
                    [p_b_hat, dist_logical, probs, first_deri_l, second_deri_l, Dgn, accuracy_record] = update_p_b_hat_logistic_NoC(...
                        temp_mat_Z, temp_Z, vcf_b_hat, threshold, lam_vec(q), a, 1, nc, d, nv, v1, v2, v3, e1, e2, e3, ie1, TRI, 30, n, 1);
                    converged_or_not(q) = dist_logical; preds = cal_probs(temp_mat_Z * p_b_hat);
                    preds_valid = setdiff(setdiff(1:1:n, find(temp_Z==1 & preds==1)), find(temp_Z==0 & preds==0)) ;
                    bic(q, 1) =  -2*sum(temp_Z(preds_valid).*log(preds(preds_valid))) - 2*sum((1-temp_Z(preds_valid)).*log(1-preds(preds_valid))) + log(length(preds_valid)) * sum(p_b_hat ~= 0);
                end
                [temp_min, temp_index] = min(bic(:, 1)); lambda_records(loop, vcf_no) = lam_vec(temp_index); optimized_lambda(vcf_no)= lam_vec(temp_index);
                [p_b_hat, dist_logical, probs, first_deri_l, second_deri_l, Dgn, accuracy_record] = update_p_b_hat_logistic_NoC(...
                    temp_mat_Z, temp_Z, vcf_b_hat, threshold, lam_vec(temp_index), a, 1, nc, d, nv, v1, v2, v3, e1, e2, e3, ie1, TRI, 30, n, 1);
                [res1, res2] = p_b_hat2TRI_NO(p_b_hat, TRI, 1); records(loop, (vcf_no-1)*nt+1:vcf_no*nt) = res1;
            end
            
        end
    end
end

if bootstrap_opt==3
    for loop=1:kLoopTime
        % Resampling by re-generating responses
        temp_preds = cal_probs(mat_Z * p_b_hat); temp_Z = binornd(1, temp_preds); 
        temp_mat_Z = mat_Z;
        [temp_b_hat dev stats] = glmfit(temp_mat_Z, temp_Z, 'binomial', 'link', 'logit', 'constant', 'off');
        
        
        if lambda_opt == 1
            % This option directly applies the optimized_lambda from original dataset
            lambda_records(loop, :) = optimized_lambda;
            [p_b_hat, dist_logical, probs, first_deri_l, second_deri_l, Dgn, accuracy_record] = update_p_b_hat_logistic_NoC(...
                temp_mat_Z, temp_Z, temp_b_hat, threshold, optimized_lambda, a, m, nc, d, nv, v1, v2, v3, e1, e2, e3, ie1, TRI, 30, n, 1);
            [res1, res2] = p_b_hat2TRI_NO(p_b_hat, TRI, m); records(loop, :) = res1;  
            
        elseif lambda_opt == 0
            % This option would find the optimized lambda
            nlam = 8; a = 3.7;threshold = 10 ^ (-3); lam_vec = linspace(0.55, 0.9, nlam);
            bic = zeros(nlam, 1); converged_or_not = zeros(nlam, 1);
            for q = 1:nlam
                [p_b_hat, dist_logical, probs, first_deri_l, second_deri_l, Dgn, accuracy_record] = update_p_b_hat_logistic_NoC(...
                    temp_mat_Z, temp_Z, temp_b_hat, threshold, lam_vec(q), a, m, nc, d, nv, v1, v2, v3, e1, e2, e3, ie1, TRI, 30, n, 1);
                converged_or_not(q) = dist_logical; preds = cal_probs(temp_mat_Z * p_b_hat);

                preds_valid = setdiff(setdiff(1:1:n, find(temp_Z==1 & preds==1)), find(temp_Z==0 & preds==0)) ;
                bic(q, 1) =  -2*sum(temp_Z(preds_valid).*log(preds(preds_valid))) - 2*sum((1-temp_Z(preds_valid)).*log(1-preds(preds_valid))) + log(length(preds_valid)) * sum(p_b_hat ~= 0);
            end

            [temp_min, temp_index] = min(bic(:, 1)); lambda_records(loop, :) = lam_vec(temp_index); 
            [p_b_hat, dist_logical, probs, first_deri_l, second_deri_l, Dgn, accuracy_record] = update_p_b_hat_logistic_NoC(...
                temp_mat_Z, temp_Z, temp_b_hat, threshold, lam_vec(temp_index), a, m, nc, d, nv, v1, v2, v3, e1, e2, e3, ie1, TRI, 30, n, 1);
            [res1, res2] = p_b_hat2TRI_NO(p_b_hat, TRI, m); records(loop, :) = res1;  
        end
         
    end
end

end