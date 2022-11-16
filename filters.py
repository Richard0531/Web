import django_filters
from .models import sampleid, MethylationProbe, DMP, DMR,lc50,discovery,screen,drugs,gene_drug
from django.db import models
from django import forms
from django_filters.widgets import RangeWidget,CSVWidget,DateRangeWidget

class SampleFilter(django_filters.FilterSet):
    class Meta:
        model = sampleid
        fields = ['pcgpid','institution']       

class ProbeFilter(django_filters.FilterSet):
    class Meta:
        model = MethylationProbe
        fields = ['chromosome','gene','name']
class DMPFilter(django_filters.FilterSet):
    class Meta:
        model = DMP
        fields = ['chromosome','gene','name']
class DMRFilter(django_filters.FilterSet):
    class Meta:
        model = DMR
        fields = ['chromosome','overlapping_genes']
class LC50Filter(django_filters.FilterSet):
    class Meta:
        model = lc50
        fields = ['pcgpids','age_group','lineage','drug','subtype_clean']
class DiscoveryFilter(django_filters.FilterSet):
    class Meta:
        model = discovery
        fields = ['geneSymbol','funcType','chromosome_name']
class ScreenFilter(django_filters.FilterSet):
    
    class Meta:
        model = screen
        fields = ['name','Index','drugs']
        filter_overrides = {
           models.CharField: {
                'filter_class': django_filters.CharFilter,
                'extra': lambda f: {
                    'lookup_expr': 'icontains',
                },
            },
        }
class DrugFilter(django_filters.FilterSet):
    drugs =django_filters.ModelMultipleChoiceFilter(queryset=drugs.objects.all(),widget=forms.CheckboxSelectMultiple)
    class Meta:
        model = screen
        fields = ['name','Index']
