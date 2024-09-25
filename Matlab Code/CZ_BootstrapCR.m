function [sep_count, cover_count, width_records, LBM_outloop] = CZ_BootstrapCR(seed, m_star_sep, m_star_s1, m_star_s2, TRI, n, nc, vx, vy, d, nv, v1, v2, v3, e1, e2, e3, ie1, m, kLoopTime, option1, option2, optionLU, a, a_tune, outloopTimes)
f = @(x,y) sqrt((x-0.5).^2 + (y-0.5).^2);fd = @(p) ddiff(drectangle(p, 0, 2, 0, 2), dcircle(p, 1, 1, 0.5));fh=@(p) 1+4*dcircle(p,1,1,0.5);
f1 = @(x,y) 1.* sin(2.*pi./(sqrt(2)-0.5) .* (sqrt((x-1).^2+(y-1).^2)-0.5))+1; f2 = @(x,y) 2.* (exp((y-1).* (y>=1)) - 1) ; f3 = @(x,y) 0 .* y;


rng(seed); nt = size(TRI, 1);

cover_count = 0; sep_count = zeros(outloopTimes, m);
bound_diff = zeros(outloopTimes, 2);LBM_outloop = cell(outloopTimes, 2); LBMUBM_outloop = cell(outloopTimes, 2);tic; width_records=zeros(outloopTimes,m);

for outloop=1:outloopTimes
   
    S = zeros(n, 1); T = zeros(n, 1); counter = 0; %(S,T) is the sample point location 
    while counter < n
        temp_r = 0.5 + betarnd(1,3,1,1) * (sqrt(2)-0.5); % Use beta distribution to generate observations
        temp_theta = 2*pi*rand(1);
        temp_S = temp_r * cos(temp_theta) + 1; temp_T = temp_r * sin(temp_theta) + 1;
        if 0 <= temp_S && temp_S <= 2 && 0<= temp_T && temp_T<= 2
            counter = counter + 1; S(counter, 1) = temp_S; T(counter, 1) = temp_T;
        end
    end

    X_1 = randn(n,1);X_2 = randn(n,1);X_3 = randn(n,1); epsilon=randn(n,1); % Generate covariates and random errors 
    beta_1 = f1(S, T); beta_2 = f2(S, T); beta_3 = f3(S, T);Z_no_epi = X_1.*beta_1 + X_2.*beta_2+X_3.*beta_3; 
    Z = Z_no_epi+epsilon.*1;

    [B, valid_id] = CZ_SPL_est(S,T,vx,vy,TRI,v1,v2,v3,nt,nc,nv,d); 

    mat_Z = zeros(n, m*nc);
    for k = valid_id
        temp1 = (B(k, :).* X_1(k,1));temp2 = (B(k, :).* X_2(k,1));temp3 = (B(k, :).* X_3(k,1));
        temp = [temp1, temp2, temp3];mat_Z(k,:) = temp;
    end
        
    mat_Z = mat_Z(valid_id, :); Z = Z(valid_id);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    mat_Z = mat_Z * (1/0.36); %%
    b_hat = (transpose(mat_Z) * mat_Z + 1 / (log(length(valid_id))*nt) * eye(m*nc)) \ transpose(mat_Z) * Z; 

    records = CZ_bootstrap_sep(0, TRI, mat_Z, Z, length(valid_id), nc, d, nv, v1, v2, v3, e1, e2, e3, ie1, m, kLoopTime, option1, option2, a, a_tune); %option2=3 to implement smaller a
    
    MCB_records = cell(size(records,1)+1, 2*m); CR_records = zeros(size(records,2)/m+1, m);
    for i = 1:m
    TRI_no = 1+(i-1)*nt:i*nt; [out, pi_idx] = sort(sum(records(:, TRI_no), 1), 'descend');
   
        for w = 0:1:nt
            cr = 0; LBM = []; UBM = [];
            for k = 0:1:nt-w
                temp_LBM = pi_idx(1:k); temp_UBM = sort(pi_idx(1:k+w));
                temp_cr = CZ_CoverageRate(records(:, TRI_no), temp_LBM, temp_UBM);
                if temp_cr > cr
                    cr = temp_cr; LBM = temp_LBM; UBM = temp_UBM; 
                end
            end
            MCB_records{w+1, 2*i-1} = sort(LBM); MCB_records{w+1, 2*i} = sort(UBM); CR_records(w+1, i) = cr;
        end
    end

    width_records(outloop,1)= find(CR_records(:,1) > 0.95, 1)-1; width_records(outloop,2)= find(CR_records(:,2) > 0.95, 1)-1; width_records(outloop,3)= find(CR_records(:,3) > 0.95, 1)-1;
    
    
    for i = 1:m
        LBM = sort(MCB_records{find(CR_records(:,i) > 0.95,1), 2*i-1}) ; UBM = sort(MCB_records{find(CR_records(:,i) > 0.95,1), 2*i}) ;
        
        if optionLU == 1
            if all(ismember(LBM, m_star_sep{i,1})) && all(ismember(m_star_sep{i,1}, UBM))
                sep_count(outloop, i) = 1;
            %elseif all(ismember(LBM, m_star_sep_alter{i,1})) && all(ismember(m_star_sep_alter{i,1}, UBM))
            %    sep_count(outloop, i) = 1;
            end
        elseif optionLU == 2
            if all(ismember(LBM, m_star_s1{i,1})) && all(ismember(m_star_s2{i,1}, UBM))
                sep_count(outloop, i) = 1;
            end
        end
        
    end
    
    LBM = [sort(MCB_records{find(CR_records(:,1) > 0.95,1), 2*1-1}) nt+sort(MCB_records{find(CR_records(:,2) > 0.95,1), 2*2-1}) 2*nt+sort(MCB_records{find(CR_records(:,3) > 0.95,1), 2*3-1})]; 
    UBM = [sort(MCB_records{find(CR_records(:,1) > 0.95,1), 2*1}) nt+sort(MCB_records{find(CR_records(:,2) > 0.95,1), 2*2}) 2*nt+sort(MCB_records{find(CR_records(:,3) > 0.95,1), 2*3})];
    
    LBMUBM_outloop{outloop, 1} = LBM; LBMUBM_outloop{outloop, 2} = UBM; m_star = [m_star_sep{2,1}+nt 2*nt+1:3*nt];
    bound_diff(outloop,1) = length(LBM) - sum(ismember(LBM, m_star)); bound_diff(outloop, 2) = length(m_star) - sum(ismember(m_star, UBM));
    LBM_outloop{outloop,1}=setdiff(LBM, m_star, 'stable') - nt;LBM_outloop{outloop,2}=setdiff(m_star, UBM, 'stable') - nt;
    
    if optionLU == 1
        if all(ismember(LBM, m_star)) && all(ismember(m_star, UBM))
            cover_count = cover_count + 1;
        %elseif all(ismember(LBM, m_star_alter)) && all(ismember(m_star_alter, UBM))
        %    cover_count = cover_count + 1;
        end
    elseif optionLU == 2
        if all(ismember(LBM, [m_star_s1{2,1}+nt 2*nt+1:3*nt])) && all(ismember([m_star_s2{2,1}+nt 2*nt+1:3*nt], UBM))
            cover_count = cover_count + 1; 
        end
        LBM_outloop{outloop,1}=setdiff(LBM, [m_star_s1{2,1}+nt 2*nt+1:3*nt], 'stable') - nt;LBM_outloop{outloop,2}=setdiff([m_star_s2{2,1}+nt 2*nt+1:3*nt], UBM, 'stable') - nt;
    end
     
    disp(['Iteration: ', num2str(outloop), ' finished and accumulative sum: ', num2str(sum(sep_count))])
end

end