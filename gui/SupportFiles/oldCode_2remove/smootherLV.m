%Smooths the curve phi, can be used several times as smoothed as you want
out=in;
for i=2:255
    out(i)=mean([in(i-1) in(i) in(i) in(i+1)]);
end
