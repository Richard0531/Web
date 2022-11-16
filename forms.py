from django import forms
from catalog.choices import *
from .models import * 
from django_select2 import forms as s2forms
from . import models
from dal import autocomplete
from .fields import ListTextWidget

class InstitutionForm(forms.Form):
    institution_name = forms.CharField(required = False, help_text = "Enter a institution name of samples")
class DMRForm(forms.Form):
    chromosome = forms.IntegerField(required = False,help_text = "Enter chromosome of DMR")
class DMPForm(forms.Form):
    chromosome = forms.IntegerField(required = False,help_text = "Enter chromosome of DMP")
class RNAForm(forms.Form):
    model = sample_information
    field = ['groups']
    gene = forms.CharField(required = False)
    drug = forms.CharField(required = False) 
    subtype = forms.CharField(required = False)
    groups = forms.CharField(required=False)
    def __init__(self, *args, **kwargs):
      _sample_list = kwargs.pop('data_list', None)
      super(RNAForm, self).__init__(*args, **kwargs)

    # the "name" parameter will allow you to use the same widget more than once in the same
    # form, not setting this parameter differently will cuse all inputs display the
    # same list.
      self.fields['groups'].widget = ListTextWidget(data_list=_sample_list, name='sample-list')
class SamplesForm(forms.Form):
    model = sample_information
    field = ['groups']
    group = forms.ModelChoiceField(queryset=model.objects.all(),to_field_name="groups", required = False)


class GroupWidget(s2forms.ModelSelect2Widget):
    search_fields = [
        'groups'
    ]
class SampleForm(forms.Form):
    model = sample_information
    field = ['groups']
    gene= forms.CharField(required=True)
    drug= forms.CharField(required=True)
    subtype = forms.CharField(required=True)
    group = forms.CharField(required=True)
    def __init__(self, *args, **kwargs):
      _sample_list = kwargs.pop('data_list', None)
      super(FormForm, self).__init__(*args, **kwargs)
      self.fields['gene'].widget = ListTextWidget(data_list=_sample_list, name='sample-list')
      self.fields['drug'].widget = ListTextWidget(data_list=_sample_list, name='sample-list')
      self.fields['subtype'].widget = ListTextWidget(data_list=_sample_list, name='sample-list')
      self.fields['group'].widget = ListTextWidget(data_list=_sample_list, name='sample-list')
class genericForm(forms.Form):
    model = sample_information
    field = ['groups']
    x_axis = forms.CharField(required = True )
    y_axis = forms.CharField(required = True )
    group = forms.ModelChoiceField(queryset=model.objects.all(),to_field_name="groups", required = False) 


class FormForm(forms.Form):
   model = sample_information
   field = ['groups']
   x_axis = forms.CharField(required=False)
   y_axis = forms.CharField(required=False)
   groups = forms.CharField(required=False)
   groups2 = forms.CharField(required=False)
   drugs = forms.CharField(required=False) 
   def __init__(self, *args, **kwargs):
      _sample_list = kwargs.pop('data_list', None)
      super(FormForm, self).__init__(*args, **kwargs)

    # the "name" parameter will allow you to use the same widget more than once in the same
    # form, not setting this parameter differently will cuse all inputs display the
    # same list.
      self.fields['x_axis'].widget = ListTextWidget(data_list=_sample_list, name='sample-list')
      self.fields['y_axis'].widget = ListTextWidget(data_list=_sample_list, name='sample-list')
      self.fields['groups'].widget = ListTextWidget(data_list=_sample_list, name='sample-list')
      self.fields['groups2'].widget = ListTextWidget(data_list=_sample_list, name='sample-list')
class FormForm2(forms.Form):
   model = sample_information
   field = ['groups']
   x_axis = forms.CharField(required=False)
   y_axis = forms.CharField(required=False)
   groups = forms.CharField(required=False)
   def __init__(self, *args, **kwargs):
      _sample_list = kwargs.pop('data_list', None)
      super(FormForm2, self).__init__(*args, **kwargs)

    # the "name" parameter will allow you to use the same widget more than once in the same
    # form, not setting this parameter differently will cuse all inputs display the
    # same list.
      self.fields['x_axis'].widget = ListTextWidget(data_list=_sample_list, name='sample-list')
      self.fields['y_axis'].widget = ListTextWidget(data_list=_sample_list, name='sample-list')
      self.fields['groups'].widget = ListTextWidget(data_list=_sample_list, name='sample-list')


