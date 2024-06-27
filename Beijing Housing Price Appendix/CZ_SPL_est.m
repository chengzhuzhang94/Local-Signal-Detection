function [B, valid_id] = CZ_SPL_est(X,Y,x,y,TRI,v1,v2,v3,nt,nc,nv,d)

[tnum,b]=tsearchn([x,y],TRI,[X,Y]);
n_x=length(X);

valid_id = 1:n_x;
B=zeros(n_x,nc);
for i=1:n_x
    j=tnum(i);
    if( ~isnan(j))
        B(i,[v1(j),v2(j),v3(j)])=basis(d,b(i,1),b(i,2),b(i,3));
    else
        valid_id = setdiff(valid_id, i);
    end
        
   
    %B(i,[v1(j),v2(j),v3(j)])=basis(d,b(i,1),b(i,2),b(i,3));
end

%B1 = B(tnum==1,:);B2 = B(tnum==2,:);




end