function [new_b_hat, dist_logical] = update_p_b_hat_poisson(mat_Z, Z, b_hat, threshold, lambda, a, m, nc, d, nv, v1, v2, v3, e1, e2, e3, ie1, TRI, kLoop, n)

nt = length(v1); temp_b_hat = b_hat; % Use temp_b_hat to record updated estimated coefficietns recursively
zero_index = ones(m*nc,1); tri_index = ones(m*nt,1); full_tri = 1:(m*nt); % Indicate non-zero estimates and valid triangles
distance = 10; % d will record the mean Euclidean distance between old b_hat and new_b_hat. If it's small enough, then the loop stops
looptime = 0;
dist_logical = false; % Boolean variabel indicating the algorithm converged or not
probs = exp(mat_Z*b_hat) ./ (1+exp(mat_Z*b_hat));

% One-time calculation. Relative triangulation part;
c_TRI = zeros(nt, 3); % Each row records the ordors of three vertices;
for j=1:nt
	c_TRI(j, :) = getindex(d,j,nv,v1,v2,v3,e1,e2,e3,ie1);
end
nc_tri = zeros(nt, nc);
for j = 1:nc
	nc_tri(:,j) = sum(c_TRI'==j); % Record vertices fall in what triangles;
end

while( distance >= 10^(-4) && looptime < kLoop)
    looptime = looptime + 1;
    % Shrink tiny estimates to 0s
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
    Dgn_lambda = zeros(sum(zero_index), sum(zero_index));count = 0;
    for g = 1:m
        for j = 1:nc
            if zero_index((g-1)*nc+j)==1
                count = count + 1;
                temp_col = nc_tri(:, j); cell =0;
                for k = 1:nt
                    if temp_col(k)==1
                        temp_b1 = temp_b_hat(nc*(g-1)+TRI(k,1));temp_b2 = temp_b_hat(nc*(g-1)+TRI(k,2));temp_b3 = temp_b_hat(nc*(g-1)+TRI(k,3));
                        temp_norm = sqrt(temp_b1^2+temp_b2^2+temp_b3^2);
                        cell = cell + (lambda*((temp_norm<=lambda)+(temp_norm>lambda) * ((a*lambda-temp_norm)>=0) * (a*lambda-temp_norm)/ (lambda*(a-1)))) / temp_norm;
                    end
                end
                Dgn_lambda(count,count) = cell;
            end
            
        end
        
    end
    %Dgn_lambda(count+1, count+1) = temp_b_hat(length(temp_b_hat));
    
    % Old part for logistic regression
    
    exp_part = exp(mat_Z(:,logical(zero_index)) * temp_b_hat(logical(zero_index)));
    first_deri_l = transpose(mat_Z(:,logical(zero_index)))*(Z-exp_part);
    second_deri_l = -transpose(mat_Z(:,logical(zero_index))) * diag(exp_part) * mat_Z(:,logical(zero_index));
    %probs_diag = diag(probs);
    %first_deri_l = transpose(mat_Z(:,logical(zero_index)))*(Z-probs);
    %second_deri_l = -transpose(mat_Z(:,logical(zero_index)))*probs_diag*(eye(length(probs))-probs_diag)*mat_Z(:,logical(zero_index));
    
    new_nonzero_b_hat = temp_b_hat(logical(zero_index)) - (-second_deri_l+length(Z).*Dgn_lambda + 10 / (log(length(Z))*nt) * eye(count)) \ (-first_deri_l+length(Z).*Dgn_lambda*temp_b_hat(logical(zero_index)));
    
    %new_nonzero_b_hat = temp_b_hat(logical(zero_index)) - (second_deri_l+length(Z).*Dgn_lambda + 10 / (log(length(Z))*nt) * eye(count)) \ (first_deri_l+length(Z).*Dgn_lambda*temp_b_hat(logical(zero_index)));
    %new_nonzero_b_hat = temp_b_hat(logical(zero_index)) - (second_deri_l+length(Z).*Dgn_lambda) \ (first_deri_l+length(Z).*Dgn_lambda*temp_b_hat(logical(zero_index)));
    new_b_hat = nonzero_to_whole(new_nonzero_b_hat, zero_index);    
    distance = mean((new_b_hat - temp_b_hat).^2); temp_b_hat = new_b_hat;
end
if distance < 10 ^ (-4)
    dist_logical = true;
end


end