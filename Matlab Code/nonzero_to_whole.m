function whole_b_hat = nonzero_to_whole(nonzero_b_hat,zero_index)
% In the zero_index, i-th 0 means the i-th element is 0, j-th 1 means the
% j-th element is non-zero
    if length(nonzero_b_hat) == sum(zero_index)
        n = length(zero_index); count = 1;
        whole_b_hat = zeros(n, 1);
        for i = 1:n
            if zero_index(i) == 0
                whole_b_hat(i) = 0;
            else
                whole_b_hat(i) = nonzero_b_hat(count); count = count + 1; % What's this count for?
            end
        end
    else
        whole_b_hat = zeros(length(zero_index), 1);
    end

end