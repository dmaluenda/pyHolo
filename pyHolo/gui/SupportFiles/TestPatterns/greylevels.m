for i=0:255
gray=i;
im=ones(1024,768)'*gray;
imwrite(im/255,[num2str(gray) 'level.png'],'PNG')
end