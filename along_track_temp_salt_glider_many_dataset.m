%% Glider track as a function of time

%clear all;

%% User input

% Glider data location

% Golf of Mexico
lon_lim = [-98 -78];
lat_lim = [18 32];
gdata = 'https://data.ioos.us/thredds/dodsC/deployments/rutgers/ng288-20180801T0000/ng288-20180801T0000.nc3.nc';
%gdata = 'https://data.ioos.us/thredds/dodsC/deployments/rutgers/ng261-20180801T0000/ng261-20180801T0000.nc3.nc';

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
date_ini = '09-Oct-2018 00:00:00';
date_end = '12-Oct-2018 00:00:00';

% GOFS3.1 outout model location
catalog31 = 'http://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_93.0/ts3z';

% Bathymetry data
%bath_data = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/GEBCO_2014_2D_-100.0_0.0_-60.0_45.0.nc';
bath_data = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/nc_files/GEBCO_2014_2D_-100.0_0.0_-10.0_70.0.nc';

% Folder where to save figure
%folder = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/';
fold = '/Volumes/aristizabal/public_html/MARACOOS_proj/Figures/Model_glider_comp/';

url = 'https://data.ioos.us/thredds/dodsC/deployments/';

id_list = char({'ng288-20180801T0000',...
           'ng261-20180801T0000',...
           'ng257-20180801T0000',...
           'ng290-20180701T0000',...
           'ng230-20180801T0000',...
           'ng258-20180801T0000',...
           'ng279-20180801T0000',...
           'ng429-20180701T0000',...
           'ru30-20180705T1825',...
           'ru33-20180801T1323',...
           'ru28-20180920T1334',...
           'blue-20180806T1400',...
           'pelagia-20180910T0000',...
           'ramses-20180704T0000',...
           'ramses-20180907T0000',...
           });   
       
  for l=1%2:8%:size(id_list,1)
    clear time_mat time_vec depth_vec temp_vec
    clear target_lon target_lat temp_gridded salt_gridded
    if l < 9
       folder = [fold,'Michael/'];
       lon_lim = [-98 -78];
       lat_lim = [18 32];
       gdata = [url,'rutgers/',strtrim(id_list(l,:)),'/',strtrim(id_list(l,:)),'.nc3.nc'];
    else
       %folder = [fold,'/MAB_SAB/'];
       lon_lim = [-81 -70];
       lat_lim = [30 42];
       gdata = [url,'secoora/',id_list(l,:),'/',id_list(l,:),'.nc3.nc'];
    end

%% Glider Extract

%ncdisp(gdata)

inst_id = ncreadatt(gdata,'/','id');
plat_type = ncreadatt(gdata,'/','platform_type');
inst = strsplit(inst_id,'-');
inst_name = inst{1};

temperature = double(ncread(gdata,'temperature'));
salinity = double(ncread(gdata,'salinity'));
pressure = double(ncread(gdata,'pressure'));
ptime = double(ncread(gdata,'precise_time'));
ptime = datenum(1970,01,01,0,0,ptime);
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

okG20 = find(depth31 == 20.0);
okG10 = find(depth31 == 10.0);

target_temp31_20m(length(oktime31))=nan;
target_temp31_10m(length(oktime31))=nan;
for i=1:length(oklon31)
    disp(i)
    %target_temp31_20m(i) = squeeze(double(ncread(catalog31,'water_temp',[oklon31(i) oklat31(i) okG20 oktime31(i)],[1 1 1 1])));
    target_temp31_10m(i) = squeeze(double(ncread(catalog31,'water_temp',[oklon31(i) oklat31(i) okG10 oktime31(i)],[1 1 1 1])));
end


%% Mean profiles

pres_gridded = 0:0.5:max(max(pressure(:,ok_time_glider)));

presok = pressure(:,ok_time_glider);
tempok = temperature(:,ok_time_glider);
temp_gridded(length(pres_gridded),size(presok,2)) = nan;

saltok = salinity(:,ok_time_glider);
salt_gridded(length(pres_gridded),size(presok,2)) = nan;

for i=1:size(pressure(:,ok_time_glider),2)
    [presu,oku] = unique(presok(:,i));
    tempu = tempok(oku,i);
    saltu = saltok(oku,i);
    ok = isfinite(tempu);
    temp_gridded(:,i) = interp1(presu(ok),tempu(ok),pres_gridded);
    salt_gridded(:,i) = interp1(presu(ok),saltu(ok),pres_gridded);
end

tempgl_mean = nanmean(temp_gridded,2);
saltgl_mean = nanmean(salt_gridded,2);

% Getting rid off profiles with no data below 100
% meters
temp_full = [];
oktt = [];
for t=1:length(timeg)
    okt = isfinite(temp_gridded(:,t));
    if sum(pres_gridded(okt) > 100) > 10
       temp_full = [temp_full temp_gridded(:,t)];
       oktt = [oktt t];
    end
end


%% Bathymetry data

%disp(bath_data);
bath_lat = ncread(bath_data,'lat');
bath_lon = ncread(bath_data,'lon');
bath_elev = ncread(bath_data,'elevation');

%% Figure glider track
%{
siz_title = 20;
siz_text = 16;

color = colormap(jet(length(latg)+1));
norm = (timeg-timeg(1))./(timeg(end)-timeg(1));
pos = round(norm.*length(latg))+1;

figure

%set(gca,'position', [0.15 0.83 0.7181 0.1243])
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

Fig_name = [folder,'glider_track_',inst_name,datestr(timeg(1),'mm-dd'),'-',datestr(timeg(end),'mm-dd')];
wysiwyg
print([Fig_name,'.png'],'-dpng','-r300') 
%}
%% Figure temp and salinity

time_mat = repmat(timeg,1,size(tempg,1))';
time_vec = reshape(time_mat,1,size(time_mat,1)*size(time_mat,2));

depth_vec = reshape(presg,1,size(presg,1)*size(presg,2));
tempg_vec = reshape(tempg,1,size(tempg,1)*size(tempg,2));
saltg_vec = reshape(saltg,1,size(saltg,1)*size(saltg,2));

marker.MarkerSize = 16;
siz_text = 16;
siz_title =16;

okt26 = tempg_vec > 25.9 & tempg_vec < 26.1;
temp26 = tempg_vec(okt26);
t26 = time_vec(okt26);
dep26 = depth_vec(okt26);

figure
set(gcf,'position',[366 82 1134 902])

subplot(211)
fast_scatter(time_vec',-depth_vec',tempg_vec','colorbar','vert','marker',marker);
hold on
plot(t26,-dep26,'.k')
set(gca,'fontsize',siz_text)
ylabel('Depth (m)')
xlabel('')
title(['Along track temperature profile ',inst_name],'fontsize',siz_title)
c = colorbar;
%c.Label.String = 'Potential Temperature (^oC)';
%c.Label.FontSize = siz_text;
colormap('jet')
%caxis(floor([min(min(tempg)) max(max(tempg))]))
%cc_vec= unique(floor(min(min(tempg)):(max(max(tempg))-min(min(tempg)))/5:max(max(tempg))));
%set(c,'ytick',cc_vec)
caxis([22 30])
xlim([timeg(1) timeg(end)])
tt_vec = unique([timeg(1),timeg(1)+(timeg(end)-timeg(1))/10:(timeg(end)-timeg(1))/10:timeg(end),timeg(end)]);
xticks(tt_vec)
xticklabels(datestr(tt_vec,'mm/dd/yy'))
%set(gca,'xticklabel',{[]})
set(gca,'TickDir','out') 
ylim([-max(depth_vec) 0])
yticks(floor(-max(depth_vec):max(depth_vec)/5:0))
%set(gca,'xgrid','on','ygrid','on','layer','top','color','k')
set(gca,'xgrid','on','ygrid','on','layer','top')

subplot(212)
fast_scatter(time_vec',-depth_vec',saltg_vec','colorbar','vert','marker',marker);
set(gca,'fontsize',siz_text)
ylabel('Depth (m)')
xlabel('')
title(['Along track salinity profile ',inst_name],'fontsize',siz_title)
c = colorbar;
%c.Label.String = 'Potential Temperature (^oC)';
%c.Label.FontSize = siz_text;
colormap('jet')
caxis([floor(min(min(saltg))) ceil(max(max(saltg)))])
cc_vec= unique(floor(min(min(saltg))):(max(max(saltg))-min(min(saltg)))/5:ceil(max(max(saltg))));
set(c,'ytick',cc_vec)
xlim([timeg(1) timeg(end)])
tt_vec = unique([timeg(1),timeg(1)+(timeg(end)-timeg(1))/10:(timeg(end)-timeg(1))/10:timeg(end),timeg(end)]);
xticks(tt_vec)
xticklabels(datestr(tt_vec,'mm/dd/yy'))
%set(gca,'xticklabel',{[]})
set(gca,'TickDir','out') 
ylim([-max(depth_vec) 0])
yticks(floor(-max(depth_vec):max(depth_vec)/5:0))
%set(gca,'xgrid','on','ygrid','on','layer','top','color','k')
set(gca,'xgrid','on','ygrid','on','layer','top')

% Figure name
Fig_name = [folder,'Along_track_temp_salt_prof_',inst_name,'_',datestr(timeg(1),'mm-dd'),'-',datestr(timeg(end),'mm-dd')];
wysiwyg
print([Fig_name,'.png'],'-dpng','-r300') 

%% Vertical profile temp using contourf

marker.MarkerSize = 16;
siz_text = 20;
siz_title =16;
cc_vec = floor(min(min(tempg))):1:ceil(max(max(tempg)));
    
figure
set(gcf,'position',[607 282 1227 703])
cc = jet(length(cc_vec)-1);
colormap(cc)
contourf(timeg(oktt),-pres_gridded,temp_full,[17:30],'.--k')
hold on
contour(timeg(oktt),-pres_gridded,temp_full,[26 26],'-k','linewidth',4)
shading interp
yvec = -max(max(presg)):0;
xvec = repmat(datenum(2018,10,10,6,0,0),1,length(yvec));
plot(xvec,yvec,'.-k','linewidth',4)
set(gca,'fontsize',siz_text)
yl = ylabel('Depth (meters)');
set(yl,'position',[datenum(2018,10,8,20,30,0) -111 0])

title(['Hurricane Michael: Glider Temperature Profiles ',inst_name],'fontsize',24)
c = colorbar;
c.Label.String = 'Sea Water Temperature (^oC)';
c.Label.FontSize = siz_text;
set(c,'YTick',cc_vec)

colormap('jet')
caxis([floor(min(min(tempg))) ceil(max(max(tempg)))])

%set(c,'ytick',cc_vec)
%xlim([timeg(1) timeg(end)])
xlim([datenum(2018,10,09) datenum(2018,10,11)])
%%tt_vec = unique([timeg(1),timeg(1)+(timeg(end)-timeg(1))/10:(timeg(end)-timeg(1))/10:timeg(end),timeg(end)]);
tt_vec = datenum(2018,10,09,00,00,00):datenum(0,0,0,6,0,0):datenum(2018,10,11,0,0,0); 
xticks(tt_vec)
xticklabels(datestr(tt_vec,'dd-HH:MM'))
xl = xlabel('2018 October Date-Time (DD-HH:MM UTC)');
%xlpos = get(xl,'position');
%set(xl,'position',[xlpos(1) xlpos(2)-7 xlpos(3)])

%set(gca,'xticklabel',{[]})
set(gca,'TickDir','out') 
ylim([-max(max(presg)) 0])
%yticks(floor(-max(max(presg)):max(max(presg))/5:0))
yticks([-200:50:0])
%set(gca,'xgrid','on','ygrid','on','layer','top','color','k')
set(gca,'xgrid','on','ygrid','on','layer','top')
%xtickangle(45)

ax = gca
ax.GridAlpha = 0.4 ;

% Figure name
Fig_name = [folder,'glider_temp_prof_',inst_name,'_',datestr(timeg(1),'mm-dd'),'-',datestr(timeg(end),'mm-dd')];
wysiwyg
print([Fig_name,'.png'],'-dpng','-r300') 

%% Temperature time series of temp at 20 m

marker.MarkerSize = 16;
siz_text = 20;
siz_title =16;

okg20 = find(pres_gridded == 20.0);
okG20 = find(depth31 == 20.0);

figure
set(gcf,'position',[607 282 1227 703])
yvec = 27.5:0.1:29.5;
xvec = repmat(datenum(2018,10,10,6,0,0),1,length(yvec));
plot(xvec,yvec,'.-k','linewidth',4)
hold on
xvec = repmat(datenum(2018,10,10,12,0,0),1,length(yvec));
plot(xvec,yvec,'--k','linewidth',4)
xvec = repmat(datenum(2018,10,09,12,0,0),1,length(yvec));
plot(xvec,yvec,'--k','linewidth',4)
h1 = plot(timeg(oktt),temp_full(okg20,:),'^-b','linewidth',4,'markersize',8,'markerfacecolor','b');
h2 = plot(time31(oktime31),target_temp31_20m,'^-r','linewidth',4,'markersize',8,'markerfacecolor','r');
set(gca,'fontsize',siz_text)
legend([h1 h2],{['Glider ',inst_name],'GOFS 3.1'},'fontsize',siz_text)
xlim([datenum(2018,10,09) datenum(2018,10,11)])
tt_vec = datenum(2018,10,09,00,00,00):datenum(0,0,0,6,0,0):datenum(2018,10,11,0,0,0); 
xticks(tt_vec)
xticklabels(datestr(tt_vec,'dd-HH:MM'))
xl = xlabel('2018 October Date-Time (DD-HH:MM UTC)');

ylabel('Sea Water Temperature (^oC)')
title('Time Series of Water Temperature at 20 Meters Depth','fontsize',30) 
ylim([27.5 29.5])
yticks([27.5:0.5:30])
grid on
ax = gca
ax.GridAlpha = 0.4 ;
ax.GridLineStyle = '--';

%%
Fig_name = [folder,'Temperature/','time_ser_temp_20m_',inst_name,'_','GOFS31_',datestr(timeg(1),'mm-dd'),'-',datestr(timeg(end),'mm-dd')];
wysiwyg
print([Fig_name,'.png'],'-dpng','-r300') 

%% Temperature time series of temp at 10 m

okg10 = find(pres_gridded == 10.0);
okG10 = find(depth31 == 10.0);

figure
set(gcf,'position',[607 282 1227 703])
yvec = 27.5:0.1:29.5;
xvec = repmat(datenum(2018,10,10,6,0,0),1,length(yvec));
plot(xvec,yvec,'.-k','linewidth',4)
hold on
xvec = repmat(datenum(2018,10,10,12,0,0),1,length(yvec));
plot(xvec,yvec,'--k','linewidth',4)
xvec = repmat(datenum(2018,10,09,12,0,0),1,length(yvec));
plot(xvec,yvec,'--k','linewidth',4)
h1 = plot(timeg(oktt),temp_full(okg10,:),'^-b','linewidth',4,'markersize',8,'markerfacecolor','b');
h2 = plot(time31(oktime31),target_temp31_10m,'^-r','linewidth',4,'markersize',8,'markerfacecolor','r');
set(gca,'fontsize',siz_text)
legend([h1 h2],{['Glider ',inst_name],'GOFS 3.1'},'fontsize',siz_text)
xlim([datenum(2018,10,09) datenum(2018,10,11)])
tt_vec = datenum(2018,10,09,00,00,00):datenum(0,0,0,6,0,0):datenum(2018,10,11,0,0,0); 
xticks(tt_vec)
xticklabels(datestr(tt_vec,'dd-HH:MM'))
xl = xlabel('2018 October Date-Time (DD-HH:MM UTC)');

ylabel('Sea Water Temperature (^oC)')
title('Time Series of Water Temperature at 10 Meters Depth','fontsize',30) 
ylim([27.5 29.5])
yticks([27.5:0.5:30])
grid on
ax = gca
ax.GridAlpha = 0.4 ;
ax.GridLineStyle = '--';

Fig_name = [folder,'time_ser_temp_10m_',inst_name,'_','GOFS31_',datestr(timeg(1),'mm-dd'),'-',datestr(timeg(end),'mm-dd')];
wysiwyg
print([Fig_name,'.png'],'-dpng','-r300') 

end

