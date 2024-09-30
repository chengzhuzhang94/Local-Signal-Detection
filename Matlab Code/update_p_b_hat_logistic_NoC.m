function [new_b_hat, dist_logical, distance_records,probs, first_deri_l, second_deri_l, Dgn_lambda, accuracy_record] = update_p_b_hat_logistic_NoC(...
    mat_Z, Z, b_hat, threshold, lambda, a, m, nc, d, nv, v1, v2, v3, e1, e2, e3, ie1, TRI, kLoop, n, option,tune)
% Purpose: update penalized estimator for logistic regression
% This function takes design matrix mat_Z, response Z and TRI info as inputs
% The output contains penalized estimator, convergence indicator and zero triangle indicator vector

% ARGUMENTS:
% mat_Z: design matrix
% Z: response variable
% b_hat: unpenalized estimator
% threshold: the threshold value for shrunking coefficients on a triangle to 0
% lambda: hyperparameter in SCAD penalty
% a: hyperparameter in SCAD penalty
% m: number of varying coefficients
% (nc,d,v1,v2,v3,nt,nv,ie1,TRI): these are associated values of Triangulation TRI
% kLoop: number of iteration
% n: sample size
% option: 1 means no tuning on diagonal matrix to avoid matrix inversion singularity, 2 means adding a small identity matrix to avoid matrix inversion singularity
% tune: adjust value that are added to avoid matrix inversion singularity

% The original INTERCEPT is also approximated by bivariate spline
distance_records = zeros(1,kLoop);
nt = length(v1); temp_b_hat = b_hat; % Use temp_b_hat to record updated estimated coefficietns recursively
zero_index = ones(m*nc,1); tri_index = ones(m*nt,1); full_tri = 1:(m*nt); % Indicate non-zero estimates and valid triangles
distance = 10; % d will record the mean Euclidean distance between old b_hat and new_b_hat. If it's small enough, then the loop stops
looptime = 0;
dist_logical = false; % Boolean variabel indicating the algorithm converged or not
probs = exp(mat_Z*b_hat) ./ (1+exp(mat_Z*b_hat)); 

% One-time calculation. Relative triangulation part;
c_TRI = zeros(nt, 3); % Each row records the orders of three vertices;
for j=1:nt
	c_TRI(j, :) = getindex(d,j,nv,v1,v2,v3,e1,e2,e3,ie1);
end
nc_tri = zeros(nt, nc);
for j = 1:nc
	nc_tri(:,j) = sum(c_TRI'==j); % Record vertices fall in what triangles;
end

accuracy_record = zeros(kLoop, 1);
while( distance >= 10^(-3) && looptime < kLoop)
    looptime = looptime + 1;
    % Shrink tiny estimates to 0s if their norms are smaller than threshold
    for i = full_tri(logical(tri_index))
        j = rem(i, nt); 
        if j == 0
            j = nt;
        end
        g = (i - j) / nt;
        n1 = g*nc+TRI(j, 1); n2 = g*nc+TRI(j, 2); n3 = g*nc+TRI(j, 3);
        temp_norm = sqrt(temp_b_hat(n1) ^ 2 + temp_b_hat(n2) ^ 2 + temp_b_hat(n3) ^ 2);
        if temp_norm < threshold
            temp_b_hat([n1, n2, n3]) = 0; zero_index([n1, n2, n3]) = 0; 
            tri_index(i) = 0;
        end
    end

    Dgn_lambda = zeros(sum(zero_index), sum(zero_index)); count = 0;
    for g = 1:m
        for j = 1:nc
            % current: the g-th varying coef and j-th basis
            if zero_index((g-1)*nc+j)==1
                count = count + 1;
                temp_col = nc_tri(:, j); cell =0;
                for k = 1:nt
                    if temp_col(k)==1
                        temp_b1 = temp_b_hat(nc*(g-1)+TRI(k,1)); temp_b2 = temp_b_hat(nc*(g-1)+TRI(k,2)); temp_b3 = temp_b_hat(nc*(g-1)+TRI(k,3));
                        temp_norm = sqrt(temp_b1^2+temp_b2^2+temp_b3^2);
                        % calculate the penalty term for the k-th triangle
                        cell = cell + (lambda*((temp_norm<=lambda)+(temp_norm>lambda) * ((a*lambda-temp_norm)>=0) * (a*lambda-temp_norm)/ (lambda*(a-1)))) / temp_norm;
                        cell(isnan(cell)) = 0;
                    end
                end
                Dgn_lambda(count,count) = cell;
            end
            
        end
        
    end
    
    probs = cal_probs(mat_Z(:,logical(zero_index))*temp_b_hat(logical(zero_index))) ;
    probs(probs >= 0.9999) = 1; probs(probs <= 0.0001) = 0;

    accuracy_record(looptime, 1) = ((sum(Z == 0 & probs < 0.5) + sum(Z == 1 & probs >= 0.5)) / n) ;
    probs_diag = diag(probs);
    first_deri_l = -transpose(mat_Z(:,logical(zero_index)))*(Z-probs);
    second_deri_l = transpose(mat_Z(:,logical(zero_index)))*probs_diag*(eye(length(probs))-probs_diag)*mat_Z(:,logical(zero_index));
    
    % Updated non-zero part estimate via the likelihood function
    if option == 2       
        % Add a diagonal term with scaling to avoid singularity
        new_nonzero_b_hat = temp_b_hat(logical(zero_index)) - (second_deri_l+length(Z).*Dgn_lambda+ tune / (log(length(Z))*nt) * eye(count)) \ (first_deri_l+length(Z).*Dgn_lambda*temp_b_hat(logical(zero_index)));
    elseif option == 1
        % Add no diagonal terms
        new_nonzero_b_hat = temp_b_hat(logical(zero_index)) - (second_deri_l+length(Z).*Dgn_lambda) \ (first_deri_l+length(Z).*Dgn_lambda*temp_b_hat(logical(zero_index)));
    elseif option == 3
        % Add a simple diagonal term withour scaling to avoid singularity
        new_nonzero_b_hat = temp_b_hat(logical(zero_index)) - (second_deri_l+length(Z).*Dgn_lambda+ tune * eye(count)) \ (first_deri_l+length(Z).*Dgn_lambda*temp_b_hat(logical(zero_index))); % Also need to set threshhold = 0
    end
    new_b_hat = nonzero_to_whole(new_nonzero_b_hat, zero_index); sum(zero_index) ;   
    distance = mean((new_b_hat - temp_b_hat).^2); temp_b_hat = new_b_hat; distance_records(looptime)=distance;
end
if distance < 10 ^ (-3)
    % determine whether convergence happened
    dist_logical = true;
end


end