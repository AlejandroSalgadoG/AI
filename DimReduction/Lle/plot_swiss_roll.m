function plot_swiss_roll(x, y, z, colors, range)   
    figure
    scatter3(x,y,z,36,colors,'filled')
    rotate3d on
    axis( repmat(range, 1, 3) )
end