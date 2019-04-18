clear;
close all;

% modify the path so it points to where readff is
addpath '/Users/aristizabal/Desktop/MARACOOS_project/COAMPS-TC/COAMPS-TC_matlab';

expdir={'14L'};
no_exp=length(expdir);
%kpath3=['/DATA/chen/TC/realtracks/uncoupled/2009'];

nest='1';
tcid='14L';
day=08;
dtg='2018100912';
subsample=10;

m=1601;
n=801;
nesto='1';
mo=num2str(m,'%04d');
no=num2str(n,'%04d');
sst(1:m,1:n,1:2)=0;
%i1=200; i2=400; j1=150; j2=350;
%i1=300; i2=550; j1=300; j2=550;
i1=1; i2=m; j1=1; j2=n;
%i1=1; i2=m; j1=1; j2=n;
for ii=1:no_exp

kpath='/Volumes/aristizabal/COAMSPA-TC/';
outpath='/Users/aristizabal/Desktop/MARACOOS_project/COAMPS-TC/COAMPS-TC_figures/';

fname=['grdlon_sfc_000000_000000_' nesto 'o' mo 'x' no '_' dtg '_00000000_datafld'];
lono=readff(kpath,fname,1)-360;

fname=['grdlat_sfc_000000_000000_' nesto 'o' mo 'x' no '_' dtg '_00000000_datafld'];
lato=readff(kpath,fname,1);

fname=['lndsea_sfc_000000_000000_' nesto 'o' mo 'x' no '_' dtg '_00000000_datafld'];

mask=readff(kpath,fname,1);
landpoints=find(mask==0);
lakepoints=find(mask==-1);

slon=lono(j1:subsample:j2,i1:subsample:i2);
slat=lato(j1:subsample:j2,i1:subsample:i2);

% tau 0 seatemp
fname=['seatmp_sfc_000000_000000_' nesto 'o' mo 'x' no '_' dtg '_00000000_fcstfld'];
sst0=readff(kpath,fname,1);

inithour=0;
ic=0;
ifover=0;
for t=6:6:120
  ic=ic+1;
hour=inithour+t;
hhour=hour;

str1=num2str(day,'%02d');
%str2=num2str(hhour,'%02d');
str2=num2str(hour,'%03d');

% read sst
fname=['seatmp_sfc_000000_000000_' nesto 'o' mo 'x' no '_' dtg '_0' str2 '0000_fcstfld'];
sst1=readff(kpath,fname,1);

if ifover == 1
fname=['uucurr_sfc_000000_000000_' nesto 'o' mo 'x' no '_' dtg '_0' str2 '0000_fcstfld'];
u=readff(kpath,fname,1);

fname=['vvcurr_sfc_000000_000000_' nesto 'o' mo 'x' no '_' dtg '_0' str2 '0000_fcstfld'];
v=readff(kpath,fname,1);

fname=['stresu_sfc_000000_000000_' nesto 'o' mo 'x' no '_' dtg '_0' str2 '0000_fcstfld'];
stresu=readff(kpath,fname,1);
fname=['stresv_sfc_000000_000000_' nesto 'o' mo 'x' no '_' dtg '_0' str2 '0000_fcstfld'];
stresv=readff(kpath,fname,1);

wstres=sqrt(stresu.^2+stresv.^2);

su=u(j1:subsample:j2,i1:subsample:i2);
sv=v(j1:subsample:j2,i1:subsample:i2);

end  % ifover

%sst(landpoints)=nan;
%sst(lakepoints)=nan;

sst1(sst1==0)=nan;
sst0(sst0==0)=nan;


% insert your variable
figure;
contourf(lono(j1:j2,i1:i2),lato(j1:j2,i1:i2),sst1(j1:j2,i1:i2),20:1:32);
%set(h,'ShowText','on','TextStep',get(h,'LevelStep')*1);
%caxis([-3 2]);
caxis([24 32]);
colorbar;                
hold all;
ic2=6*(ic-1)+1;
%if ic2 > 41
%plot(lon_fcst(1:6:36),lat_fcst(1:6:36),'color',[0.4 0 0],'MarkerFaceColor',[0.4 0 0],'marker','o','markersize',3,'markeredgecolor',[1 0 0], 'LineWidth',2)
%else
%plot(lon_fcst(1:6:ic2),lat_fcst(1:6:ic2),'color',[0.4 0 0],'MarkerFaceColor',[0.4 0 0],'marker','o','markersize',3,'markeredgecolor',[1 0 0], 'LineWidth',2)
%end


xlabel('Longitude');
ylabel('Latitude');
titlestring=[dtg ' ' num2str(hour,'%02d') ' h fcst SST'];
title(titlestring)

box on
fname=[outpath 'sst_' num2str(t,'%02d') '.png'];
print('-dpng',fname)

figure;
%clf


% dsst

dsst=sst1-sst0;
exp(ii).sst0=sst0;
exp(ii).sst=sst1;
exp(ii).dsst=dsst;
[C,h]=contourf(lono(j1:j2,i1:i2),lato(j1:j2,i1:i2),dsst(j1:j2,i1:i2),-6:1:6);
caxis([-3 2])
%pcolor(lono(j1:j2,i1:i2),lato(j1:j2,i1:i2),dsst(j1:j2,i1:i2));shading interp;colorbar
colorbar;
hold all;


xlabel('Longitude');
ylabel('Latitude');
titlestring=[dtg ' ' num2str(hour,'%02d') 'h fcst dsst, min:' num2str(min(min(dsst(j1:j2,i1:i2))),'%4.1f')];
title(titlestring)
fname=[outpath 'dsst_' num2str(t,'%02d') '.png'];
print(fname,'-dpng','-r300') 

end % no_exp loop
end