%%
clear all

%% User input

% ng288 (Golf of Mexico)
lon_lim = [-98 -78];
lat_lim = [18 32];
gdata = 'https://data.ioos.us/thredds/dodsC/deployments/rutgers/ng288-20180801T0000/ng288-20180801T0000.nc3.nc';

% Folder where to save figure
folder = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/';

%%
% ng288
url_glider = gdata;
model_name = 'GOFS 3.1';
url_model = 'http://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_93.0/ts3z';
var = 'temperature';
fig = 'no';
date_ini = '05-Oct-2018 00:00:00';
date_end = '13-Oct-2018 03:00:00';

[timeg_vec, depthg_vec, varg_matrix,timem,depthm,varm] = ...
          glider_transect_model_comp(url_glider,model_name,url_model,var,fig,date_ini,date_end);

%%  Temperature 

% Getting rid off profiles with no data below 100
% meters
var_full = [];
oktt = [];
for t=1:length(timeg_vec)
    okt = isfinite(varg_matrix(:,t));
    if sum(depthg_vec(okt) > 100) > 10
       var_full = [var_full varg_matrix(:,t)];
       oktt = [oktt t];
    end
end

siz_text = 20;

var_name = ncreadatt(url_glider,var,'ioos_category');
var_units = ncreadatt(url_glider,var,'units');

inst_id = ncreadatt(url_glider,'/','id');
inst = strsplit(inst_id,'-');
inst_name = inst{1};

%cc_vec = floor(min(min(varg_matrix))):1:ceil(max(max(varg_matrix)));
cc_vec = 17:30; 

figure
set(gcf,'position',[327 434 1301 521*2])

subplot(3,2,1)
set(gca,'position',[0.08 0.7093 0.4 0.2157])
contourf(timeg_vec(oktt),-depthg_vec,var_full,cc_vec,'.--k')
hold on
contour(timeg_vec,-depthg_vec,varg_matrix,[26 26],'-k','linewidth',2)
shading interp

yvec = -max(max(depthg_vec)):0;
xvec = repmat(datenum(2018,10,10,6,0,0),1,length(yvec));
plot(xvec,yvec,'.-k','linewidth',4)

set(gca,'fontsize',siz_text)
ylabel('Depth (m)')
title(['Along Track ',var_name,' ',inst_name],'fontsize',24)

cc = jet(length(cc_vec)-1);
colormap(cc)
%c = colorbar;
%c.Label.String = [var_name,' ','(',var_units,')'];
%c.Label.FontSize = siz_text;
%caxis([floor(min(min(varg_matrix))) ceil(max(max(varg_matrix)))])
caxis([17 30])
%set(c,'ytick',cc_vec)

tt_vec = unique(floor([timeg_vec(1),timeg_vec(1)+(timeg_vec(end)-timeg_vec(1))/10:(timeg_vec(end)-timeg_vec(1))/10:timeg_vec(end),timeg_vec(end)]));
xticks(tt_vec)
xticklabels(datestr(tt_vec,'dd'))
%xl = xlabel('October 2018');
%xticklabels(datestr(tt_vec,'dd-HH:MM'))
%xl = xlabel('2018 October Date-Time (DD-HH:MM UTC)');

%tt_vec = unique(floor([timeg_vec(1),timeg_vec(1)+(timeg_vec(end)-timeg_vec(1))/10:(timeg_vec(end)-timeg_vec(1))/10:timeg_vec(end),timeg_vec(end)]));
%xticks(tt_vec)
%xticklabels(datestr(tt_vec,'mm/dd/yy'))
%xlim([tt_vec(1) timeg_vec(end)])

%ylim([-max(depthg_vec) 0])
%yticks(floor(-max(depthg_vec):max(depthg_vec)/5:0))
ylim([-250 0])
yticks(-250:50:0)


%patch([datenum(2018,8,21,9,0,0) datenum(2018,8,21,12,0,0) datenum(2018,8,21,12,0,0) datenum(2018,8,21,9,0,0)],...
%      [-300 -300 0 0],[0.5 0.5 0.5],'facealpha',0.3) 
  
%patch([datenum(2018,8,22,9,0,0) datenum(2018,8,22,12,0,0) datenum(2018,8,22,12,0,0) datenum(2018,8,22,9,0,0)],...
%      [-300 -300 0 0],[0.5 0.5 0.5],'facealpha',0.3)   

set(gca,'TickDir','out') 
set(gca,'xgrid','on','ygrid','on','layer','top')

ax = gca;
ax.GridAlpha = 0.3;

%% Vertical profile temp using contourf
%{
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
%}
%%
subplot(3,2,2)
set(gca,'position',[0.51 0.7093 0.4 0.2157])
contourf(timem(2:end),-depthm,varm(:,2:end),cc_vec,'.--k')
hold on
contour(timem(2:end),-depthm,varm(:,2:end),[26 26],'-k','linewidth',2)
shading interp

yvec = -max(max(depthg_vec)):0;
xvec = repmat(datenum(2018,10,10,6,0,0),1,length(yvec));
plot(xvec,yvec,'.-k','linewidth',4)

set(gca,'fontsize',24)
ylabel('Depth (m)')
title(['Along Track ',var_name,' ',inst_name],'fontsize',24)

cc = jet(length(cc_vec)-1);
colormap(cc)
c = colorbar;
%c.Label.String = [var_name,' ','(',var_units,')'];
c.Label.String = '(^oC)';
c.Label.FontSize = 20;
%caxis([floor(min(min(varg_matrix))) ceil(max(max(varg_matrix)))])
caxis([17 30])
set(c,'position',[0.92 0.7094 0.0123 0.2156])
set(c,'ticks',17:2:31)
%set(c,'ytick',cc_vec)

yticks(-250:50:0)
ylim([-250 0])

tt_vec = unique(floor([timeg_vec(1),timeg_vec(1)+(timeg_vec(end)-timeg_vec(1))/10:(timeg_vec(end)-timeg_vec(1))/10:timeg_vec(end),timeg_vec(end)]));
xticks(tt_vec)
xticklabels(datestr(tt_vec,'dd'))
%xl = xlabel('October 2018');

set(gca,'fontsize',siz_text)
%ylabel('Depth (m)')
ylabel(' ')
set(gca,'ytick',[])
title(['Along track ',var_name,' ',model_name],'fontsize',24)

%{
cc = jet(length(cc_vec)-1);
colormap(cc)
c = colorbar;
c.Label.String = [var_name,' ','(',var_units,')'];
c.Label.FontSize = siz_text;
caxis([floor(min(min(varg_matrix))) ceil(max(max(varg_matrix)))])
set(c,'ytick',cc_vec)

tt_vec = unique(floor([timeg_vec(1),timeg_vec(1)+(timeg_vec(end)-timeg_vec(1))/10:(timeg_vec(end)-timeg_vec(1))/10:timeg_vec(end),timeg_vec(end)]));
xticks(tt_vec)
xticklabels(datestr(tt_vec,'mm/dd/yy'))
xlim([tt_vec(1) timeg_vec(end)])
%}

%ylim([-max(depthg_vec) 0])
%yticks(floor(-max(depthg_vec):max(depthg_vec)/5:0))


%patch([datenum(2018,8,21,9,0,0) datenum(2018,8,21,12,0,0) datenum(2018,8,21,12,0,0) datenum(2018,8,21,9,0,0)],...
%      [-300 -300 0 0],[0.5 0.5 0.5],'facealpha',0.3) 
  
%patch([datenum(2018,8,22,9,0,0) datenum(2018,8,22,12,0,0) datenum(2018,8,22,12,0,0) datenum(2018,8,22,9,0,0)],...
%      [-300 -300 0 0],[0.5 0.5 0.5],'facealpha',0.3)  

set(gca,'TickDir','out') 
set(gca,'xgrid','on','ygrid','on','layer','top')

ax = gca;
ax.GridAlpha = 0.3;      
      
% Figure name
%Fig_name = [folder,'Along_track_',var,'_prof_',inst_name,'_',datestr(timeg(1),'mm-dd-yy'),'-',datestr(timeg(end),'mm-dd-yy')];
wysiwyg
%print([Fig_name,'.png'],'-dpng','-r300') 

%% Temperature time series of temp at 10 m

marker.MarkerSize = 16;
siz_text = 20;
siz_title =16;

okg10 = find(depthg_vec == 10.0);
okG10 = find(depthm == 10.0);

subplot(3,2,[3,4])
set(gca,'position',[0.0800 0.4111 0.8624 0.2142])
yvec = 27.5:0.1:29.5;
xvec = repmat(datenum(2018,10,10,6,0,0),1,length(yvec));
plot(xvec,yvec,'.-k','linewidth',4)
hold on

dep = 10;
pos31 = interp1(depthm,1:length(depthm),dep);
posglider = interp1(depthg_vec,1:length(depthg_vec),dep);

%yvec = 23:0.1:30.5;
%xvec = repmat(datenum(2018,10,10,6,0,0),1,length(yvec));
%plot(xvec,yvec,'.-k','linewidth',4)
ymin1 = round(min(varg_matrix(posglider,:)))-1;
ymin2 = round(min(varm(pos31,:)))-1;
ymin = min([ymin1,ymin2]);
ymax1 = round(max(varg_matrix(posglider,:)))+1;
ymax2 = round(max(varm(pos31,:)))+1;
ymax = max([ymax1,ymax2]);
y = [ymin ymin ymax ymax];
x = [datenum(2018,10,05,09,0,0) datenum(2018,10,05,12,0,0) datenum(2018,10,05,12,0,0) datenum(2018,10,05,09,0,0)];
patch(x,y,[0.9 0.9 0.9])
x = [datenum(2018,10,06,09,0,0) datenum(2018,10,06,12,0,0) datenum(2018,10,06,12,0,0) datenum(2018,10,06,09,0,0)];
patch(x,y,[0.9 0.9 0.9])
x = [datenum(2018,10,07,09,0,0) datenum(2018,10,07,12,0,0) datenum(2018,10,07,12,0,0) datenum(2018,10,07,09,0,0)];
patch(x,y,[0.9 0.9 0.9])
x = [datenum(2018,10,08,09,0,0) datenum(2018,10,08,12,0,0) datenum(2018,10,08,12,0,0) datenum(2018,10,08,09,0,0)];
patch(x,y,[0.9 0.9 0.9])
x = [datenum(2018,10,09,09,0,0) datenum(2018,10,09,12,0,0) datenum(2018,10,09,12,0,0) datenum(2018,10,09,09,0,0)];
patch(x,y,[0.9 0.9 0.9])
x = [datenum(2018,10,10,09,0,0) datenum(2018,10,10,12,0,0) datenum(2018,10,10,12,0,0) datenum(2018,10,10,09,0,0)];
patch(x,y,[0.9 0.9 0.9])
x = [datenum(2018,10,11,09,0,0) datenum(2018,10,11,12,0,0) datenum(2018,10,11,12,0,0) datenum(2018,10,11,09,0,0)];
patch(x,y,[0.9 0.9 0.9])
x = [datenum(2018,10,12,09,0,0) datenum(2018,10,12,12,0,0) datenum(2018,10,12,12,0,0) datenum(2018,10,12,09,0,0)];
patch(x,y,[0.9 0.9 0.9])

%xvec = repmat(datenum(2018,10,10,12,0,0),1,length(yvec));
%plot(xvec,yvec,'--k','linewidth',4)
%xvec = repmat(datenum(2018,10,09,12,0,0),1,length(yvec));
plot(xvec,yvec,'--k','linewidth',4)
h1 = plot(timeg_vec(oktt),var_full(okg10,:),'^-b','linewidth',4,'markersize',8,'markerfacecolor','b');
h2 = plot(timem,varm(depthm==10,:),'^-r','linewidth',4,'markersize',8,'markerfacecolor','r');
set(gca,'fontsize',siz_text)
legend([h1 h2],{['Glider ',inst_name],'GOFS 3.1'},'fontsize',siz_text)
xlim([datenum(2018,10,05) datenum(2018,10,13)])
%tt_vec = datenum(2018,10,09,00,00,00):datenum(0,0,0,6,0,0):datenum(2018,10,11,0,0,0); 
%xticks(tt_vec)
xticklabels(datestr(tt_vec,'dd'))
%xl = xlabel('October 2018');

yl = ylabel('(^oC)');
set(yl,'position',[datenum(2018,10,04,14,0,0) 28.5 0])
title('Time Series of Water Temperature at 10 Meters Depth','fontsize',24) 
ylim([27.5 29.5])
yticks(27.5:0.5:30)
grid on
ax = gca;
ax.GridAlpha = 0.4 ;
ax.GridLineStyle = '--';

%% Temperature time series of temp at 100 m

marker.MarkerSize = 16;
siz_text = 20;
siz_title =16;

dep = 100.0;
%pos31 = interp1(depthm,1:length(depthm),dep);
%posglider = interp1(depthg_vec,1:length(depthg_vec),dep);
okg = find(depthg_vec == dep);
okG = find(depthm == dep);


subplot(3,2,[5,6])
set(gca,'position',[0.08  0.1100 0.8624 0.2157])
%set(gcf,'position',[607 282 1227 703])
%yvec = 27.5:0.1:29.5;
%xvec = repmat(datenum(2018,10,10,6,0,0),1,length(yvec));
%plot(xvec,yvec,'.-k','linewidth',4)
hold on

yvec = 22:0.1:28;
xvec = repmat(datenum(2018,10,10,6,0,0),1,length(yvec));
plot(xvec,yvec,'.-k','linewidth',4)
ymin1 = round(min(varg_matrix(okg,:)))-1;
ymin2 = round(min(varm(okG,:)))-1;
ymin = min([ymin1,ymin2]);
ymax1 = round(max(varg_matrix(okg,:)))+1;
ymax2 = round(max(varm(okG,:)))+1;
ymax = max([ymax1,ymax2]);
y = [ymin ymin ymax ymax];
x = [datenum(2018,10,05,09,0,0) datenum(2018,10,05,12,0,0) datenum(2018,10,05,12,0,0) datenum(2018,10,05,09,0,0)];
patch(x,y,[0.9 0.9 0.9])
x = [datenum(2018,10,06,09,0,0) datenum(2018,10,06,12,0,0) datenum(2018,10,06,12,0,0) datenum(2018,10,06,09,0,0)];
patch(x,y,[0.9 0.9 0.9])
x = [datenum(2018,10,07,09,0,0) datenum(2018,10,07,12,0,0) datenum(2018,10,07,12,0,0) datenum(2018,10,07,09,0,0)];
patch(x,y,[0.9 0.9 0.9])
x = [datenum(2018,10,08,09,0,0) datenum(2018,10,08,12,0,0) datenum(2018,10,08,12,0,0) datenum(2018,10,08,09,0,0)];
patch(x,y,[0.9 0.9 0.9])
x = [datenum(2018,10,09,09,0,0) datenum(2018,10,09,12,0,0) datenum(2018,10,09,12,0,0) datenum(2018,10,09,09,0,0)];
patch(x,y,[0.9 0.9 0.9])
x = [datenum(2018,10,10,09,0,0) datenum(2018,10,10,12,0,0) datenum(2018,10,10,12,0,0) datenum(2018,10,10,09,0,0)];
patch(x,y,[0.9 0.9 0.9])
x = [datenum(2018,10,11,09,0,0) datenum(2018,10,11,12,0,0) datenum(2018,10,11,12,0,0) datenum(2018,10,11,09,0,0)];
patch(x,y,[0.9 0.9 0.9])
x = [datenum(2018,10,12,09,0,0) datenum(2018,10,12,12,0,0) datenum(2018,10,12,12,0,0) datenum(2018,10,12,09,0,0)];
patch(x,y,[0.9 0.9 0.9])

%xvec = repmat(datenum(2018,10,10,12,0,0),1,length(yvec));
%plot(xvec,yvec,'--k','linewidth',4)
%xvec = repmat(datenum(2018,10,09,12,0,0),1,length(yvec));
%plot(xvec,yvec,'--k','linewidth',4)
h1 = plot(timeg_vec(oktt),var_full(okg,:),'^-b','linewidth',4,'markersize',8,'markerfacecolor','b');
h2 = plot(timem,varm(okG,:),'^-r','linewidth',4,'markersize',8,'markerfacecolor','r');
set(gca,'fontsize',siz_text)
legend([h1 h2],{['Glider ',inst_name],'GOFS 3.1'},'fontsize',siz_text)
xlim([datenum(2018,10,05) datenum(2018,10,13)])
%tt_vec = datenum(2018,10,09,00,00,00):datenum(0,0,0,6,0,0):datenum(2018,10,11,0,0,0); 
%xticks(tt_vec)
xticklabels(datestr(tt_vec,'dd'))
xl = xlabel('October 2018');

yl = ylabel('(^oC)');
set(yl,'position',[datenum(2018,10,04,14,0,0) 25 0])
title(['Time Series of Water Temperature at ',num2str(dep),' Meters Depth'],'fontsize',24) 
%ylim([27.5 29.5])
%yticks(27.5:0.5:30)
grid on
ax = gca;
ax.GridAlpha = 0.4 ;
ax.GridLineStyle = '--';

% Figure name
Fig_name = [folder,'Along_track_',var,'-',inst_name,'_',datestr(timeg_vec(1),'mm-dd-yy'),'-',datestr(timeg_vec(end),'mm-dd-yy'),'-time_series'];
wysiwyg
print([Fig_name,'.png'],'-dpng','-r300') 
