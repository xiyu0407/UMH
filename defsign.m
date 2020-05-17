function A=defsign(A)
    for i=1:size(A,1)
        for j=1:size(A,2)
            if A(i,j)<0
                A(i,j)=-1;
            elseif A(i,j)>=0
                A(i,j)=1;
            else
                error('errors in quantization.Maybe is NaN');
            end
        end
    end
end