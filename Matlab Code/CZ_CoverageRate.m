function [cr] = CZ_CoverageRate(records, LBM, UBM)
p = size(records, 2); count = 0;
for i = 1:size(records,1)
    temp_row = records(i, :); temp_idx = find(temp_row == 1);
    if all(ismember(LBM, temp_idx)) && all(ismember(temp_idx, UBM))
        count = count + 1;
    end
end
cr = count/size(records,1);

end

% records = [0 1 1; 0 1 0; 1 0 0]; LBM = [2]; UBM = [2 3]; CZ_CoverageRate(records, LBM, UBM)