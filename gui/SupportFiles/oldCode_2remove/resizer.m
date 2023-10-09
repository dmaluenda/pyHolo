function[OUT]=resizer(IN)
%resize IN of 1xN in OUT of 1xM erasing the trivial components

p=find(IN,1,'last');
OUT=IN(1:p);

dif=max(size(IN))-max(size(OUT));
if dif~=0
disp(['erazing ' num2str(dif) ' trivial components'])
end
