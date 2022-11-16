from django.urls import path
from . import views
from django.conf.urls import include
urlpatterns = [
path('', views.index, name='index'),
path('samples/',views.SampleListView.as_view(), name='samples'),
path('samples/plot', views.sample_plot, name='sample-plot'),
path('overview', views.sample_overview, name='overview'),
path('samples/plot', views.sample_plot, name='sample-plot'),
path('samples/<str:pk>', views.SampleDetailView.as_view(), name='samples-detail'),
path('probes/', views.ProbeListView.as_view(), name='probes'),
path('probes/<str:pk>', views.ProbeDetailView.as_view(), name='probes-detail'),
path('dmp/', views.DMPListView.as_view(), name='dmp'),
path('dmp/<str:pk>', views.DMPDetailView.as_view(), name='dmp-detail'),
path('dmp/plot', views.dmp_plot, name='dmp-plot'),
path('dmr/', views.DMRListView.as_view(), name='dmr'),
path('dmr/plot', views.dmr_plot, name='dmr-plot'),
path('dmr/<str:pk>', views.DMRDetailView.as_view(), name='dmr-detail'),
path('lc50/',views.Lc50ListView.as_view(), name='lc50'),
path('lc50/plot', views.lc50_plot, name='lc50-plot'),
path('discovery/', views.DiscoveryListView.as_view(), name='discovery'),
path('screen/', views.ScreenListView.as_view(), name='screens'),
path('django_plotly_dash/', include('django_plotly_dash.urls')),
path('overall', views.overall,  name='overall'),
path('drugoverview', views.sample_overview,  name='drugoverview'),
path('snv', views.snv,  name='snv'),
path('cnv', views.cnv,  name='cnv'),
]
