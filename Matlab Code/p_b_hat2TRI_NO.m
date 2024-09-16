function [added_vec, tri_index] = p_b_hat2TRI_NO(p_b_hat, TRI, m)
%% added_vec is a 1 x mNt vector, in which 1 indicates estimated null region while 0 means the other; tri_index contains all such kind of triangles' order numbers

nc = length(p_b_hat) / m; nrow = size(TRI, 1); 
added_vec = zeros(1, nrow*m); tri_index = [];
for loop = 1:m
   temp = p_b_hat(1+(loop-1)*nc:loop*nc) ; temp = find(temp==0) ;
   for row = 1:nrow
       if all(ismember(TRI(row, :), temp))
          added_vec(row+(loop-1)*nrow) = 1; 
          tri_index = [tri_index row+(loop-1)*nrow] ;
       end
   end
   
end

end

% p_b_hat = [0 0 1 0 0]; TRI = [1 2 5; 2 4 5;2 3 4]; p_b_hat2TRI_NO(p_b_hat, TRI, 1) 
% Output: [1 1 0]