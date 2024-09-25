function probs = cal_probs(x)
    x(x > 100) = 100;
    probs = exp(x) ./ (1+exp(x));
    probs(probs>0.9999) = 1;
    probs(probs<0.0001) = 0;
    
end