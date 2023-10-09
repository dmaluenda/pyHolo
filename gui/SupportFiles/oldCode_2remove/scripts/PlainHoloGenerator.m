display=1;
hologram=mapa_Holo3(1,30,[768 1024],display); %Holo=f(T,ph[º],N,SLM)
name=['ref_' num2str(display) '.bmp'];
imwrite(hologram,name,'BMP')
imshow(hologram)