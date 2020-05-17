function value =getL21norm(T)
value =0;
    for i=1 :size(T,1)
        
        value = value + norm(T(i,:),'fro');
       
    end
end