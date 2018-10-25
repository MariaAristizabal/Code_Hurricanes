%% Glider track as a function of time

clear all;

%% User input

% Glider data location

% Golf of Mexico
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

% ng300 (Virgin Islands)
%lon_lim = [-68 -64];
%lat_lim = [15 20];
%gdata = 'http://data.ioos.us/thredds/dodsC/deployments/rutgers/ng300-20180701T0000/ng300-20180701T0000.nc3.nc';

% Initial and final date
date_ini = '01-Oct-2018 00:00:00';
date_end = '12-Oct-2018 00:00:00';

% GOFS3.1 outout model location
catalog31 = 'http://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_93.0/ts3z';

% GOFS3.1 output model location
catalog30 = 'http://tds.hycom.org/thredds/dodsC/GLBu0.08/expt_91.2/ts3z';

% Bathymetry data
%bath_data = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/GEBCO_2014_2D_-100.0_0.0_-60.0_45.0.nc';
bath_data = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/nc_files/GEBCO_2014_2D_-100.0_0.0_-10.0_70.0.nc';

% Folder where to save figure
%folder = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/Temperature/';
folder = '/Volumes/aristizabal/public_html/MARACOOS_proj/Figures/Model_glider_comp/Michael/';

%% Glider Extract

inst_id = ncreadatt(gdata,'/','id');
plat_type = ncreadatt(gdata,'/','platform_type');
inst = strsplit(inst_id,'-');
inst_name = inst{1};

temperature = double(ncread(gdata,'temperature'));
salinity = double(ncread(gdata,'salinity'));
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
saltg = salinity(:,ok_time_glider);
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

target_salt31(length(depth31),length(oktime31))=nan;
for i=1:length(oklon31)
    target_salt31(:,i) = squeeze(double(ncread(catalog31,'salinity',[oklon31(i) oklat31(i) 1 oktime31(i)],[1 1 inf 1])));
end

%% GOFS 3.0

lat30 = ncread(catalog30,'lat');
lon30 = ncread(catalog30,'lon');
depth30 = ncread(catalog30,'depth');
time30 = ncread(catalog30,'time'); % hours since 2000-01-01 00:00:00
time30 = time30/24 + datenum(2000,01,01,0,0,0);

%oktime30 = find(time30 >= time(1) & time30 < time(end));
oktime30 = find(time30 >= tti & time30 < tte);

sublon30=interp1(time,target_lon,time30(oktime30));
sublat30=interp1(time,target_lat,time30(oktime30));

oklon30=round(interp1(lon30,1:length(lon30),sublon30));
oklat30=round(interp1(lat30,1:length(lat30),sublat30));

target_salt30(length(depth30),length(oktime30))=nan;
for i=1:length(oklon30)
    target_salt30(:,i) = squeeze(double(ncread(catalog30,'salinity',[oklon30(i) oklat30(i) 1 oktime30(i)],[1 1 inf 1])));
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
saltg_vec = reshape(saltg,1,size(saltg,1)*size(saltg,2));

marker.MarkerSize = 16;

figure
set(gcf,'position',[366 82 1134 902])

%%
subplot(411)
set(gca,'position', [0.15 0.83 0.7181 0.1243])
contour(bath_lon(oklonbath),bath_lat(oklatbath),bath_elev(oklonbath,oklatbath)',[0,-50,-100,-200,-1000,-2000,-4000,-8000],...
        'color','k','linewidth',2)
%contour(bath_lon,bath_lat,bath_elev',[0 0],'color','k','linewidth',2)
hold on
%contourf(bath_lon,bath_lat,bath_elev',[0 max(max(bath_elev))],'color',[0.5 0.2 0])
%contourf(bath_lon,bath_lat,bath_elev',[min(min(bath_elev)) -0.1])
axis equal
ylim(lat_lim)
xlim(lon_lim)
set(gca,'fontsize',siz_text)
t = title([inst_name,' ',' Track'],'fontsize',siz_title);
%set(t,'position',[-75.0000   38.35   -0.0000])
%set(t,'position',[-73.0000   40.6   -0.0000])

%%
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

subplot(412)
[h,c_h] = fast_scatter(time_vec',-depth_vec',saltg_vec','colorbar','vert','marker',marker);
hold on
set(gca,'fontsize',siz_text)
ylabel('Depth (m)')
xlabel('')
title(['Along track salinity profile ',inst_name],'fontsize',siz_title)
c = colorbar;
%c.Label.String = 'Potential Temperature (^oC)';
%c.Label.FontSize = siz_text;
colormap('jet')
caxis([floor(min(min(saltg))) ceil(max(max(saltg)))])
c_vec = floor([floor(min(min(saltg))):(max(max(saltg))-min(min(saltg)))/5:ceil(max(max(saltg))),ceil(max(max(saltg)))]);
cc_vec = unique(c_vec);
set(c,'ytick',cc_vec)
xlim([timeg(1) timeg(end)])
tt_vec = unique([timeg(1),timeg(1)+(timeg(end)-timeg(1))/10:(timeg(end)-timeg(1))/10:timeg(end),timeg(end)]);
xticks(tt_vec)
%datetick('x','keepticks')
set(gca,'xticklabel',{[]})
set(gca,'TickDir','out') 
ylim([-max(depth_vec) 0])
yticks(floor(-max(depth_vec):max(depth_vec)/5:0))
%set(gca,'xgrid','on','ygrid','on','layer','top','color','k')
set(gca,'xgrid','on','ygrid','on','layer','top')

%%
subplot(413)
pcolor(time31(oktime31),-depth31,target_salt31)
shading interp
set(gca,'fontsize',siz_text)
ylabel('Depth (m)')
title('GOFS 3.1','fontsize',siz_title)
c = colorbar;
c.Label.String = 'Salinity (psu)';
c.Label.FontSize = siz_text;
colormap('jet')
caxis([floor(min(min(saltg))) ceil(max(max(saltg)))])
c_vec = floor([floor(min(min(saltg))):(max(max(saltg))-min(min(saltg)))/5:ceil(max(max(saltg))),ceil(max(max(saltg)))]);
cc_vec = unique(c_vec);
set(c,'ytick',cc_vec)
xlim([timeg(1) timeg(end)])
tt_vec = unique([timeg(1),timeg(1)+(timeg(end)-timeg(1))/10:(timeg(end)-timeg(1))/10:timeg(end),timeg(end)]);
xticks(tt_vec)
%datetick('x','keepticks')
set(gca,'xticklabel',{[]})
set(gca,'TickDir','out','ytick',[]) 
ylim([-max(depth_vec) 0])
yticks(floor(-max(depth_vec):max(depth_vec)/5:0))
set(gca,'xgrid','on','ygrid','on','layer','top')

%%
subplot(414)
pcolor(time30(oktime30),-depth30,target_salt30)
shading interp
set(gca,'fontsize',siz_text)
ylabel('Depth (m)')
title('GOFS 3.0','fontsize',siz_title)
c = colorbar;
%c.Label.String = 'Potential Temperature (^oC)';
%c.Label.FontSize = siz_text;
%set(c,'ytick',min(min(temperature)):4:min(min(temperature)))
colormap('jet')
caxis([floor(min(min(saltg))) ceil(max(max(saltg)))])
c_vec = floor([floor(min(min(saltg))):(max(max(saltg))-min(min(saltg)))/5:ceil(max(max(saltg))),ceil(max(max(saltg)))]);
cc_vec = unique(c_vec);
set(c,'ytick',cc_vec)
xlim([timeg(1) timeg(end)])
tt_vec = unique([timeg(1),timeg(1)+(timeg(end)-timeg(1))/10:(timeg(end)-timeg(1))/10:timeg(end),timeg(end)]);
xticks(tt_vec)
xticklabels(datestr(tt_vec,'mm/dd/yy'))
%datetick('x','keepticks')
set(gca,'TickDir','out') 
%set(gca,'xticklabel',{[]})
ylim([-max(depth_vec) 0])
yticks(floor(-max(depth_vec):max(depth_vec)/5:0))
set(gca,'position',[0.1300    0.1100    0.735    0.15])
set(c,'position',[0.872    0.1098    0.0141    0.15])
set(gca,'xgrid','on','ygrid','on','layer','top')
xtickangle(45)

%%
% Figure name
Fig_name = [folder,'salinity/','Along_track_salt_prof_glider_2models_',inst_name,'_',datestr(timeg(1),'mm-dd-yy'),'-',datestr(timeg(end),'mm-dd-yy')];
wysiwyg
print([Fig_name,'.png'],'-dpng','-r300') 
