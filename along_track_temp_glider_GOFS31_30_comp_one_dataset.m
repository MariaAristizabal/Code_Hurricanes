%% Glider track as a function of time

clear all;

%% User input

% Glider data location

% ng288 (Golf of Mexico)
lon_lim = [-98 -78];
lat_lim = [18 32];
gdata = 'https://data.ioos.us/thredds/dodsC/deployments/rutgers/ng288-20180801T0000/ng288-20180801T0000.nc3.nc';

% RAMSES (MAB + SAB)
%lon_lim = [-81 -70];
%lat_lim = [30 42];
%gdata = 'https://data.ioos.us/thredds/dodsC/deployments/secoora/ramses-20180907T0000/ramses-20180907T0000.nc3.nc';

% RU33 (MAB + SAB)
%lon_lim = [-81 -70];
%lat_lim = [30 42];
%gdata = 'https://data.ioos.us/thredds/dodsC/deployments/rutgers/ru33-20180801T1323/ru33-20180801T1323.nc3.nc';

% RU28 (MAB + SAB)
%lon_lim = [-81 -70];
%lat_lim = [30 42];
%gdata = 'https://data.ioos.us/thredds/dodsC/deployments/rutgers/ru28-20180920T1334/ru28-20180920T1334.nc3.nc';

% blue (MAB + SAB)
%lon_lim = [-81 -70];
%lat_lim = [30 42];
%gdata = 'https://data.ioos.us/thredds/dodsC/deployments/rutgers/blue-20180806T1400/blue-20180806T1400.nc3.nc';

% RU30 (MAB + SAB)
%lon_lim = [-81 -70];
%lat_lim = [30 42];
%gdata = 'https://data.ioos.us/thredds/dodsC/deployments/rutgers/ru30-20180705T1825/ru30-20180705T1825.nc3.nc';

% Silvia (MAB + SAB)
%lon_lim = [-81 -70];
%lat_lim = [30 42];
%gdata = 'https://data.ioos.us/thredds/dodsC/deployments/rutgers/sylvia-20180802T0930/sylvia-20180802T0930.nc3.nc';

% ng300 (Virgin Islands)
%lon_lim = [-68 -64];
%lat_lim = [15 20];
%gdata = 'http://data.ioos.us/thredds/dodsC/deployments/rutgers/ng300-20180701T0000/ng300-20180701T0000.nc3.nc';

% cp_336 (MAB)
%lon_lim = [-81 -70];
%lat_lim = [30 42];
%gdata = 'https://data.ioos.us/thredds/dodsC/deployments/rutgers/cp_336-20180724T1433/cp_336-20180724T1433.nc3.nc';

% cp_376 (MAB)
%lon_lim = [-81 -70];
%lat_lim = [30 42];
%gdata = 'https://data.ioos.us/thredds/dodsC/deployments/rutgers/cp_376-20180724T1552/cp_376-20180724T1552.nc3.nc';

% cp_376 (MAB)
%lon_lim = [-81 -70];
%lat_lim = [30 42];
%gdata = 'https://data.ioos.us/thredds/dodsC/deployments/rutgers/cp_389-20180724T1620/cp_389-20180724T1620.nc3.nc';

% Initial and final date
date_ini = '06-Oct-2018 00:00:00';
date_end = '14-Oct-2018 00:00:00';
%date_ini = ''; %if empty, date_ini is the firts time stamp in data
%date_end = ''; %if empty, date_end is the last time stamp in data

% GOFS3.1 outout model location
catalog31 = 'http://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_93.0/ts3z';

% GOFS3.1 output model location
catalog30 = 'http://tds.hycom.org/thredds/dodsC/GLBu0.08/expt_91.2/ts3z';

% Bathymetry data
%bath_data = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/GEBCO_2014_2D_-100.0_0.0_-60.0_45.0.nc';
bath_data = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/nc_files/GEBCO_2014_2D_-100.0_0.0_-10.0_70.0.nc';

% Folder where to save figure
folder = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/';
%folder = '/Volumes/aristizabal/public_html/MARACOOS_proj/Figures/Model_glider_comp/Michael/';

%% Glider Extract

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
if isempty(date_ini)
   tti = time(1);
else
   tti = datenum(date_ini);
end

if isempty(date_end)
   tte = time(end);
else
   tte = datenum(date_end);
end

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
target_lon(1:length(timeg)) = nan;
for i=1:length(timeg)
    if long(i) < 0 
       target_lon(i) = 360 + long(i);
    else
       target_lon(i) = long(i);
    end
end
target_lat = latg;

sublon31=interp1(timeg,target_lon,time31(oktime31));
sublat31=interp1(timeg,target_lat,time31(oktime31));

oklon31=round(interp1(lon31,1:length(lon31),sublon31));
oklat31=round(interp1(lat31,1:length(lat31),sublat31));

target_temp31(length(depth31),length(oktime31))=nan;
for i=1:length(oklon31)
    disp(length(oklon31))
    disp(['GOFS31',' i= ',num2str(i)])
    if isfinite(oklon31(i)) && isfinite(oklat31(i)) && isfinite(oktime31(i))
       target_temp31(:,i) = squeeze(double(ncread(catalog31,'water_temp',[oklon31(i) oklat31(i) 1 oktime31(i)],[1 1 inf 1])));
    end
end

%% GOFS 3.0

lat30 = ncread(catalog30,'lat');
lon30 = ncread(catalog30,'lon');
depth30 = ncread(catalog30,'depth');
time30 = ncread(catalog30,'time'); % hours since 2000-01-01 00:00:00
time30 = time30/24 + datenum(2000,01,01,0,0,0);

%oktime30 = find(time30 >= time(1) & time30 < time(end));
oktime30 = find(time30 >= tti & time30 < tte);

sublon30=interp1(timeg,target_lon,time30(oktime30));
sublat30=interp1(timeg,target_lat,time30(oktime30));

oklon30=round(interp1(lon30,1:length(lon30),sublon30));
oklat30=round(interp1(lat30,1:length(lat30),sublat30));

target_temp30(length(depth30),length(oktime30))=nan;
for i=1:length(oklon30)
    disp(length(oklon30))
    disp(['GOFS30',' i= ',num2str(i)])
    if isfinite(oklon30(i)) && isfinite(oklat30(i)) && isfinite(oktime30(i))
       target_temp30(:,i) = squeeze(double(ncread(catalog30,'water_temp',[oklon30(i) oklat30(i) 1 oktime30(i)],[1 1 inf 1])));
    end
end


%% Bathymetry data

%disp(bath_data);
bath_lat = ncread(bath_data,'lat');
bath_lon = ncread(bath_data,'lon');
bath_elev = ncread(bath_data,'elevation');

oklatbath = bath_lat >= lat_lim(1) & bath_lat <= lat_lim(2);
oklonbath = bath_lon >= lon_lim(1) & bath_lon <= lon_lim(2);

%% Figure
%{
siz_title = 20;
siz_text = 16;

color = colormap(jet(length(latg)+1));
norm = (timeg-timeg(1))./(timeg(end)-timeg(1));
pos = round(norm.*length(latg))+1;

time_mat = repmat(timeg,1,size(tempg,1))';
time_vec = reshape(time_mat,1,size(time_mat,1)*size(time_mat,2));

depth_vec = reshape(presg,1,size(presg,1)*size(presg,2));
tempg_vec = reshape(tempg,1,size(tempg,1)*size(tempg,2));

marker.MarkerSize = 16;

% Figure name
Fig_name = [folder,'Along_track_temp_prof_glider_model',inst_name,'_',plat_type,'_'];

figure
set(gcf,'position',[366 82 1134 902])

subplot(311)
contour(bath_lon,bath_lat,bath_elev',[0,-50,-100,-200,-1000,-2000,-4000,-8000],...
        'color','k','linewidth',2)
%contour(bath_lon,bath_lat,bath_elev',[0 0],'color','k','linewidth',2)
hold on
%contourf(bath_lon,bath_lat,bath_elev',[0 max(max(bath_elev))],'color',[0.5 0.2 0])
%contourf(bath_lon,bath_lat,bath_elev',[min(min(bath_elev)) -0.1])
axis equal
ylim(lat_lim)
xlim(lon_lim)
set(gca,'fontsize',siz_text)
title([inst_name,' ',plat_type,' Track'],'fontsize',siz_title)

for i=1:length(latg)
    plot(long(i),latg(i),'.','markersize',12,'color',color(pos(i),:,:))
end
colormap('jet')
c = colorbar('v');
caxis([timeg(1) timeg(end)])
timevec = datenum(date_ini):datenum(0,0,2,0,0,0):datenum(date_end);
time_lab = datestr(timevec,'mmm/dd');
set(c,'ytick',timevec)
datetick(c,'keepticks')
set(c,'yticklabel',time_lab)

subplot(312)
fast_scatter(time_vec',-depth_vec',tempg_vec','colorbar','vert','marker',marker);
set(gca,'fontsize',siz_text)
ylabel('Depth (m)')
xlabel('')
title(['Along track temperature profile ',inst_name,' ',plat_type],'fontsize',siz_title)
c = colorbar;
c.Label.String = 'Potential Temperature (^oC)';
c.Label.FontSize = siz_text;
colormap('jet')
caxis([min(min(temperature)) max(max(temperature))])
set(c,'ytick',round(min(min(temperature))):2:round(max(max(temperature))))
xlim([datenum(date_ini) datenum(date_end)])
xticks(datenum(date_ini):datenum(0,0,2):datenum(date_end))
datetick('x','keepticks')
ylim([-max(depth_vec) 0])
grid on

subplot(313)
pcolor(time31(oktime31),-depth31,target_temp31)
shading interp
set(gca,'fontsize',siz_text)
ylabel('Depth (m)')
title(['Along track temperature profile ','GOFS 3.1'],'fontsize',siz_title)
c = colorbar;
c.Label.String = 'Potential Temperature (^oC)';
c.Label.FontSize = siz_text;
set(c,'ytick',min(min(temperature)):2:min(min(temperature)))
colormap('jet')
caxis([min(min(temperature)) max(max(temperature))])
set(c,'ytick',round(min(min(temperature))):2:round(max(max(temperature))))
xlim([datenum(date_ini) datenum(date_end)])
xticks(datenum(date_ini):datenum(0,0,2):datenum(date_end))
datetick('x','keepticks')
ylim([-max(depth_vec) 0])
grid on

wysiwyg
print([Fig_name,'.png'],'-dpng','-r300') 
%}

%% Figure with 2 models: GOFS3.1, GOFS3.0

siz_title = 20;
siz_text = 16;

color = colormap(jet(length(latg)+1));
norm = (timeg-timeg(1))./(timeg(end)-timeg(1));
pos = round(norm.*length(latg))+1;

time_mat = repmat(timeg,1,size(tempg,1))';
time_vec = reshape(time_mat,1,size(time_mat,1)*size(time_mat,2));

depth_vec = reshape(presg,1,size(presg,1)*size(presg,2));
tempg_vec = reshape(tempg,1,size(tempg,1)*size(tempg,2));

marker.MarkerSize = 16;

figure
set(gcf,'position',[366 82 1134 902])


subplot(411)
set(gca,'position', [0.38 0.78 0.19 0.19])
contour(bath_lon(oklonbath),bath_lat(oklatbath),bath_elev(oklonbath,oklatbath)',[0,-50,-100,-200,-1000,-2000,-4000,-8000],...
        'color','k','linewidth',2)
%contour(bath_lon,bath_lat,bath_elev',[0 0],'color','k','linewidth',2)
hold on
%contourf(bath_lon,bath_lat,bath_elev',[0 max(max(bath_elev))],'color',[0.5 0.2 0])
%contourf(bath_lon,bath_lat,bath_elev',[min(min(bath_elev)) -0.1])
axis equal
%xlim([-76 -72])
%ylim([37 41])
xlim(lon_lim)
ylim(lat_lim)
set(gca,'fontsize',siz_text)
t = title([inst_name,' ',' Track'],'fontsize',siz_title);
%set(t,'position',[-75.0000   38.35   -0.0000])
%set(t,'position',[-73.0000   40.6   -0.0000])

for i=1:length(latg)
    plot(long(i),latg(i),'.','markersize',12,'color',color(pos(i),:,:))
end
colormap('jet')
c = colorbar('v');
caxis([timeg(1) timeg(end)])
tt_vec = unique([timeg(1),timeg(1)+(timeg(end)-timeg(1))/5:(timeg(end)-timeg(1))/5:timeg(end),timeg(end)]);
time_lab = datestr(tt_vec,'mm/dd/yy');
set(c,'ytick',tt_vec)
datetick(c,'keepticks')
set(c,'yticklabel',time_lab)

%%
okt26 = tempg_vec > 25.9 & tempg_vec < 26.1;
temp26 = tempg_vec(okt26);
t26 = time_vec(okt26);
dep26 = depth_vec(okt26);

subplot(412)
[h,c_h] = fast_scatter(time_vec',-depth_vec',tempg_vec','colorbar','vert','marker',marker);
hold on
plot(t26,-dep26,'.k')

set(gca,'fontsize',siz_text)
ylabel('Depth (m)')
title(['Along track temperature profile ',inst_name],'fontsize',siz_title)

c = colorbar;
colormap('jet')
caxis([floor(min(tempg_vec)) ceil(max(tempg_vec))])
cc_vec = unique(round(floor(min(tempg_vec)):(max(tempg_vec)-min(tempg_vec))/5:ceil(max(tempg_vec))));
set(c,'ytick',cc_vec)

tt_vec = unique(floor([time_vec(1),time_vec(1)+(time_vec(end)-time_vec(1))/10:(time_vec(end)-time_vec(1))/10:time_vec(end),time_vec(end)]));
xticks(tt_vec)
set(gca,'xticklabel',{[]})
xlim([tt_vec(1) time_vec(end)])

ylim([-max(depth_vec) 0])
yticks(floor(-max(depth_vec):max(depth_vec)/5:0))

set(gca,'TickDir','out') 
set(gca,'xgrid','on','ygrid','on','layer','top')

ax = gca;
ax.GridAlpha = 0.3;

%%
subplot(413)
pcolor(time31(oktime31),-depth31,target_temp31)
hold on
contour(time31(oktime31),-depth31,target_temp31,[26,26],'color','k','linewidth',3)
shading interp

set(gca,'fontsize',siz_text)
ylabel('Depth (m)')
title('GOFS 3.1','fontsize',siz_title)

c = colorbar;
colormap('jet')
caxis([floor(min(tempg_vec)) ceil(max(tempg_vec))])
cc_vec = unique(round(floor(min(tempg_vec)):(max(tempg_vec)-min(tempg_vec))/5:ceil(max(tempg_vec))));
set(c,'ytick',cc_vec)
c.Label.String = 'Water Temperature (^oC)';
c.Label.FontSize = siz_text;

tt_vec = unique(floor([time_vec(1),time_vec(1)+(time_vec(end)-time_vec(1))/10:(time_vec(end)-time_vec(1))/10:time_vec(end),time_vec(end)]));
xticks(tt_vec)
set(gca,'xticklabel',{[]})
xlim([tt_vec(1) time_vec(end)])

ylim([-max(depth_vec) 0])
yticks(floor(-max(depth_vec):max(depth_vec)/5:0))

set(gca,'TickDir','out') 
set(gca,'xgrid','on','ygrid','on','layer','top')

ax = gca;
ax.GridAlpha = 0.3;

%%
subplot(414)
pcolor(time30(oktime30),-depth30,target_temp30)
hold on
contour(time30(oktime30),-depth30,target_temp30,[26,26],'color','k','linewidth',3)
shading interp

set(gca,'fontsize',siz_text)
ylabel('Depth (m)')
title('GOFS 3.0','fontsize',siz_title)

c = colorbar;
colormap('jet')
caxis([floor(min(tempg_vec)) ceil(max(tempg_vec))])
cc_vec = unique(round(floor(min(tempg_vec)):(max(tempg_vec)-min(tempg_vec))/5:ceil(max(tempg_vec))));
set(c,'ytick',cc_vec)

tt_vec = unique(floor([time_vec(1),time_vec(1)+(time_vec(end)-time_vec(1))/10:(time_vec(end)-time_vec(1))/10:time_vec(end),time_vec(end)]));
xticks(tt_vec)
xticklabels(datestr(tt_vec,'mm/dd/yy'))
xlim([tt_vec(1) time_vec(end)])
xtickangle(45)

ylim([-max(depth_vec) 0])
yticks(floor(-max(depth_vec):max(depth_vec)/5:0))

set(gca,'position',[0.1300    0.1100    0.735    0.15])
set(c,'position',[0.872    0.1098    0.0141    0.15])

set(gca,'TickDir','out') 
set(gca,'xgrid','on','ygrid','on','layer','top')

ax = gca;
ax.GridAlpha = 0.3;

%%
% Figure name
Fig_name = [folder,'Along_track_temp_prof_glider_2models_',inst_name,'_',datestr(timeg(1),'mm-dd-yy'),'-',datestr(timeg(end),'mm-dd-yy')];
wysiwyg
print([Fig_name,'.png'],'-dpng','-r300') 