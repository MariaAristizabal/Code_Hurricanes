%% Glider track as a function of time

clear all;

%% User input

% Glider data location

% RU22
lon_lim = [125 127];
lat_lim = [32 34];
gdata = 'https://data.ioos.us/thredds/dodsC/deployments/rutgers/ru22-20180815T0107/ru22-20180815T0107.nc3.nc';

% Golf of Mexico
%lon_lim = [-98 -78];
%lat_lim = [18 32];
%gdata = 'https://data.ioos.us/thredds/dodsC/deployments/rutgers/ng288-20180801T0000/ng288-20180801T0000.nc3.nc';

% RAMSES (MAB + SAB)
%lon_lim = [-81 -70];
%lat_lim = [30 42];
%gdata = 'https://data.ioos.us/thredds/dodsC/deployments/secoora/ramses-20180907T0000/ramses-20180907T0000.nc3.nc';

% RU33 (MAB + SAB)
%lon_lim = [-81 -70];
%lat_lim = [30 42];
%gdata = 'https://data.ioos.us/thredds/dodsC/deployments/rutgers/ru33-20180801T1323/ru33-20180801T1323.nc3.nc';

% ng300 (Virgin Islands)
%lon_lim = [-68 -64];
%lat_lim = [15 20];
%gdata = 'http://data.ioos.us/thredds/dodsC/deployments/rutgers/ng300-20180701T0000/ng300-20180701T0000.nc3.nc';

% Initial and final date
%date_ini = '01-Oct-2018 00:00:00';
%date_end = '11-Oct-2018 00:00:00';
date_ini = '16-Aug-2018 00:00:00';
date_end = '25-Aug-2018 00:00:00';

% GOFS3.1 outout model location
catalog31 = 'http://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_93.0/ts3z';

% Folder where to save figure
folder = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/Temperature/';
%folder = '/Volumes/aristizabal/public_html/MARACOOS_proj/Figures/Model_glider_comp/Michael/';

%% Glider Extract

%ncdisp(gdata);

inst_id = ncreadatt(gdata,'/','id');
plat_type = ncreadatt(gdata,'/','platform_type');
inst = strsplit(inst_id,'-');
inst_name = inst{1};

temperature = double(ncread(gdata,'temperature'));
pressure = double(ncread(gdata,'pressure'));
time = double(ncread(gdata,'time'));
time = datenum(1970,01,01,0,0,time);
latitude = double(ncread(gdata,'latitude'));
longitude = double(ncread(gdata,'longitude'));

% Finding subset of data for time period of interest
tti = datenum(date_ini);
tte = datenum(date_end);
ok_time_glider = find(time >= tti & time < tte);

tempg = temperature(:,ok_time_glider);
presg = pressure(:,ok_time_glider);
latg = latitude(ok_time_glider);
long = longitude(ok_time_glider);
timeg = time(ok_time_glider);

% Final lat and lon of deployment for time period of interest 
% This lat and lon is used to find the grid point in the model output
% we can consider changing it to the average lat, lon for the entire
% deployment
Glat = latitude(ok_time_glider(end));
Glon = longitude(ok_time_glider(end));

%% GOFS 3.1

%ncdisp(catalog31);

lat31 = ncread(catalog31,'lat');
lon31 = ncread(catalog31,'lon');
depth31 = ncread(catalog31,'depth');
tim31 = ncread(catalog31,'time'); % hours since 2000-01-01 00:00:00

time31 = tim31/24 + datenum(2000,01,01,0,0,0);

%oktime31 = find(time31 >= time(1) & time31 < time(end));
oktime31 = find(time31 >= tti & time31 < tte);

% Conversion from glider longitude and latitude to GOFS3.1 convention
target_lon(1:length(longitude)) = nan;
for i=1:length(time)
    if longitude(i) < 0 
       target_lon(i) = 360 + longitude(i);
    else
       target_lon(i) = longitude(i);
    end
end
target_lat = latitude;

sublon31=interp1(time,target_lon,time31(oktime31));
sublat31=interp1(time,target_lat,time31(oktime31));

oklon31=round(interp1(lon31,1:length(lon31),sublon31));
oklat31=round(interp1(lat31,1:length(lat31),sublat31));

target_temp31(length(depth31),length(oktime31))=nan;
for i=1:length(oklon31)
    disp(i)
    target_temp31(:,i) = squeeze(double(ncread(catalog31,'water_temp',[oklon31(i) oklat31(i) 1 oktime31(i)],[1 1 inf 1])));
end

%% Figure GOFS3.1

marker.MarkerSize = 16;
siz_text = 20;
siz_title =30;
cc_vec = floor(min(min(tempg))):1:ceil(max(max(tempg)));

figure
set(gcf,'position',[607 282 1227 703])

contourf(time31(oktime31),-depth31,target_temp31,17:30,'.--k')
hold on
contour(time31(oktime31),-depth31,target_temp31,[26 26],'-k','linewidth',4)
shading interp
yvec = -max(max(presg)):0;
xvec = repmat(datenum(2018,10,10,6,0,0),1,length(yvec));
plot(xvec,yvec,'.-k','linewidth',4)

set(gca,'fontsize',siz_text)
title(['Hurricane Michael: GOFS3.1 Temperature Profiles ',inst_name],'fontsize',24)
yl = ylabel('Depth (meters)');
set(yl,'position',[datenum(2018,10,8,20,30,0) -111 0])
xlim([datenum(2018,10,09) datenum(2018,10,11)])
tt_vec = datenum(2018,10,09,00,00,00):datenum(0,0,0,6,0,0):datenum(2018,10,11,0,0,0); 
xticks(tt_vec)
xticklabels(datestr(tt_vec,'dd-HH:MM'))
xl = xlabel('2018 October Date-Time (DD-HH:MM UTC)');
%xlpos = get(xl,'position');
%set(xl,'position',[xlpos(1) xlpos(2) xlpos(3)])

set(gca,'TickDir','out') 
ylim([-max(max(presg)) 0])
yticks(-200:50:0)
set(gca,'xgrid','on','ygrid','on','layer','top')

cc = jet(length(cc_vec)-1);
colormap(cc)
c = colorbar;
caxis([floor(min(min(tempg))) ceil(max(max(tempg)))])
c.Label.String = 'Sea Water Temperature (^oC)';
c.Label.FontSize = siz_text;
set(c,'YTick',cc_vec)

ax = gca;
ax.GridAlpha = 0.4 ;

% Figure name
Fig_name = [folder,'GOFS31_temp_prof_',inst_name,'_',datestr(timeg(1),'mm-dd'),'-',datestr(timeg(end),'mm-dd')];
wysiwyg
print([Fig_name,'.png'],'-dpng','-r300') 
