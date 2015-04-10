function m = ACCURACY(computed,actual,num_classes)

n = length(actual);
modes = zeros(num_classes,1);

for k=1:num_classes
   v = actual( computed == k ); 
   [~,modes(k)] = mode(v);
end

m = (sum(modes)/n)*100;