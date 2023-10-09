for display=1:2;
hologram=mapa_Holo3(0,0,[768 1024],display);
imwrite(hologram,['SwitchOff_' num2str(display) '.bmp'],'BMP')
%imshow(hologram)
end