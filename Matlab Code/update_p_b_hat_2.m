function [new_b_hat, dist_logical, tri_index] = update_p_b_hat_2(mat_Z, Z, b_hat, threshold, lambda, a, m, nc, d, nv, v1, v2, v3, e1, e2, e3, ie1, TRI, kLoop, n, option, tune)
% This function takes design matrix mat_Z, response Z and TRI info as inputs
% The output contains penalized estimator, convergence indicator and zero
% triangle indicator vector

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

nt = size(TRI,1);
temp_b_hat = b_hat; % initialize penalized estimator with the input UNPEN estimator 
zero_index = ones(m*nc,1); tri_index = ones(m*nt,1); full_tri = 1:(m*nt);
distance = 10; % it will record the mean Euclidean distance between old b_hat and new_b_hat. If it's small enough, then the loop stops
looptime = 0;
dist_logical = false;

% One-time calculation. Relative triangulation part;
c_TRI = zeros(nt, 3); % Each row records the orders of three vertices;
for j=1:nt
	c_TRI(j, :) = getindex(d,j,nv,v1,v2,v3,e1,e2,e3,ie1);
end
nc_tri = zeros(nt, nc);
for j = 1:nc
	nc_tri(:,j) = sum(c_TRI'==j); % Record vertices fall in what triangles;
end

while( distance >= 10^(-4) && looptime < kLoop)
    looptime = looptime + 1;
    
    for i = full_tri(logical(tri_index))
        j = rem(i, nt); 
        if j == 0
            j = nt;
        end
        g = (i - j) / nt;
        n1 = g*nc+TRI(j, 1); n2 = g*nc+TRI(j, 2); n3 = g*nc+TRI(j, 3);
        temp_norm = sqrt(temp_b_hat(n1) ^ 2 + temp_b_hat(n2) ^ 2 + temp_b_hat(n3) ^ 2);
        if temp_norm < threshold
            % if the norm of current 3 points associated to the same
            % triangle is smaller than the threshold, then their values are
            % shrunk to 0s
            temp_b_hat([n1, n2, n3]) = 0; zero_index([n1, n2, n3]) = 0; 
            tri_index(i) = 0;
        end
    end
    
    Dgn_lambda = zeros(sum(zero_index), sum(zero_index));count = 0;
    for g = 1:m
        for j = 1:nc
            % if the associated coefficient is still not zero
            if zero_index((g-1)*nc+j)==1
          
                count = count + 1;
                temp_col = nc_tri(:, j); cell =0;
                for k = 1:nt
                    if temp_col(k)==1
                        temp_b1 = temp_b_hat(nc*(g-1)+TRI(k,1));
                        temp_b2 = temp_b_hat(nc*(g-1)+TRI(k,2));
                        temp_b3 = temp_b_hat(nc*(g-1)+TRI(k,3));
                        temp_norm = sqrt(temp_b1^2+temp_b2^2+temp_b3^2);
                        cell = cell + (lambda*((temp_norm<=lambda)+(temp_norm>lambda) * ((a*lambda-temp_norm)>=0) * (a*lambda-temp_norm)/ (lambda*(a-1)))) / temp_norm;
                    end
                end
                Dgn_lambda(count,count) = cell;
            end
            
        end
        
    end

    % update SCAD penalized estimator
    if option == 2
        new_nonzero_b_hat = (transpose(mat_Z(:,logical(zero_index))) * mat_Z(:,logical(zero_index)) + n .* Dgn_lambda + tune / (log(length(Z))*nt) * eye(count)) \ transpose(mat_Z(:,logical(zero_index))) * Z;
    elseif option == 1
        new_nonzero_b_hat = (transpose(mat_Z(:,logical(zero_index))) * mat_Z(:,logical(zero_index)) + n .* Dgn_lambda) \ transpose(mat_Z(:,logical(zero_index))) * Z;
    end
    new_b_hat = nonzero_to_whole(new_nonzero_b_hat, zero_index); 
    
    % calculate the diff between new SCAD estimator and the previous SCAD estimator
    distance = mean((new_b_hat - temp_b_hat).^2); 
    temp_b_hat = new_b_hat;
      
end

new_b_hat(abs(new_b_hat) < 1e-3) = 0;

% update convergence info
if distance < 10 ^ (-4)
    dist_logical = true;
end

end