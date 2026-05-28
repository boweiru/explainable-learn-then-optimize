function p=drawgyroid(v,sizeofdata,nofv,nofgyroid,data0,data01)
minxyz=min(data0);
maxxyz=max(data0);
yspace=linspace(minxyz(1)-(nofv-1)/2*(nofgyroid/nofv),maxxyz(1)+(nofv-1)/2*(nofgyroid/nofv),sizeofdata(1)*nofv);
xspace=linspace(minxyz(2)-(nofv-1)/2*(nofgyroid/nofv),maxxyz(2)+(nofv-1)/2*(nofgyroid/nofv),sizeofdata(2)*nofv);
zspace=linspace(minxyz(3)-(nofv-1)/2*(nofgyroid/nofv),maxxyz(3)+(nofv-1)/2*(nofgyroid/nofv),sizeofdata(3)*nofv);
[x,y,z]=meshgrid(xspace,yspace,zspace);
o = sin(2*pi/nofgyroid*(x)).*cos(2*pi/nofgyroid*y) + sin(2*pi/nofgyroid*y).*cos(2*pi/nofgyroid*z) + sin(2*pi/nofgyroid*z).*cos(2*pi/nofgyroid*(x));

xleft= (y==(minxyz(1)-(nofv-1)/2*(nofgyroid/nofv))) ;
o(xleft) =0.9;
xright= (y==(maxxyz(1)+(nofv-1)/2*(nofgyroid/nofv))) ;
o(xright) =0.9;
yleft= (x==(minxyz(2)-(nofv-1)/2*(nofgyroid/nofv))) ;
o(yleft) =0.9;
yright= (x==(maxxyz(2)+(nofv-1)/2*(nofgyroid/nofv))) ;
o(yright) =0.9;
zleft= (z==(minxyz(3)-(nofv-1)/2*(nofgyroid/nofv)) );
o(zleft) =0.9;
zright= (z==(maxxyz(3)+(nofv-1)/2*(nofgyroid/nofv))) ;
o(zright) =0.9;

o=o+v;
p = isosurface(x,y,z,o,0.9);                     

end