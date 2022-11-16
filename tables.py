import django_tables2 as tables
from .models import *
from django_tables2.export.views import ExportMixin



class SampleTable(tables.Table):
    class Meta:
        model = sampleid
        template_name = "django_tables2/bootstrap4.html"
       
class ProbeTable(tables.Table):
    class Meta:
        model = MethylationProbe
        template_name = "django_tables2/bootstrap4.html"
class DMPTable(tables.Table):
    class Meta:
        model = DMP
        template_name = "django_tables2/bootstrap4.html"
class DMRTable(tables.Table):
    class Meta:
        model = DMR
        template_name = "django_tables2/bootstrap4.html"

class LC50Table(tables.Table):
    class Meta:
        model = lc50
        template_name = "django_tables2/bootstrap4.html"
class DiscoveryTable(tables.Table):
    class Meta:
        model = discovery
        template_name = "django_tables2/bootstrap4.html"
        
class ScreenTable(tables.Table):
    class Meta:
        model = screen
        template_name = "django_tables2/bootstrap4.html"






