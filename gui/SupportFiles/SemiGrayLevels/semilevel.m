im=zeros(768,1024)*256;
for i=0:255
    im(1:1024*768/2)=i;
    imwrite(im/255,[num2str(i) 'semilevel.png'],'PNG');
end
