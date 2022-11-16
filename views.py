from django.shortcuts import render
from .models import *
from django.contrib.auth.decorators import login_required
import pandas as pd
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
from plotly.offline import plot
import plotly.express as px
import plotly.graph_objs as go
from catalog.forms import *
from catalog.choices import *
from catalog.files import * 
import numpy as np
import dash_bio
from django_filters.views import FilterView
from django_tables2.views import SingleTableMixin
from .filters import *
from django.template import context
from django.http import HttpResponse
from django.templatetags.static import static
import csv
import os
import fastparquet 
from fastparquet import write, ParquetFile
import pyarrow.parquet as pq
from pyarrow.parquet import ParquetFile
from django.http import HttpResponseRedirect
import numpy as np
from dal import autocomplete
import json
import urllib.request as urlreq
import dash
from dash import Dash, dcc, html, Input, Output, State, MATCH, ALL, dash_table
from dash.dependencies import Input, Output
import dash_bio as dashbio
from django_plotly_dash import DjangoDash
import plotly.io as pio
import pyarrow.parquet
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
pio.kaleido.scope.default_format = "svg"
@login_required(login_url='/accounts/login/')
def index(request):
    """View function for home page of site."""

    context = {
    }

    # Render the HTML template index.html with the data in the context variable
    return render(request, 'index.html', context=context)
@login_required(login_url='/accounts/login/')
def sample_plot (request):     
    lc50 = pd.read_parquet('static/LC50_plots.parquet',engine = 'pyarrow')
    groups = request.GET.get('group')
    group = 'status'
    if groups:
        group = groups
    fig = px.histogram(
        lc50,x = group, color = group
    ,height=800
    )
    plots = plot(fig, output_type="div")


    context = {'plot_div': plots,'form': SamplesForm}
    
    return render(request, 'catalog/sample_plol.html', context=context)


@login_required(login_url='/accounts/login/')
def dmr_plot (request):


    dmr = DMR.objects.all()
    chromosomes = request.GET.get('chromosome')
   
    if chromosomes:
        dmr = dmr.filter(chromosome = 'chr'+ chromosomes)
   

    project_data_dmr = [
        {
            'Chromosome' : x.chromosome,
            'Mean Difference' : x.meandiff,
            'P': x.Fisher,
            'Gene':x.overlapping_genes
        } for x in dmr

    ]
    df_dmr = pd.DataFrame(project_data_dmr)
    df_dmr['Mean Difference'] = pd.to_numeric(df_dmr['Mean Difference'])
    df_dmr['P'] = pd.to_numeric(df_dmr['P'])
    df_dmr['logP'] = np.log10(df_dmr['P'])
    df_dmr['size'] = -(df_dmr['logP'])
    df_dmr_filtered = DMRFilter(request.GET, queryset = dmr)
    """ Figure Data"""
    fig_dmr =px.scatter (
      df_dmr, x = 'Mean Difference', y = 'logP',size = 'size',color = 'Chromosome', hover_name = 'Gene',
       height = 800, width = 1200
    )
    fig_dmr.update_yaxes(autorange="reversed")
    scatter_plot_dmr = plot(fig_dmr,output_type="div", show_link=False, link_text="")
    context = {
        'plot_div': scatter_plot_dmr,
        'form': DMRForm()
    }

    # Render the HTML template index.html with the data in the context variable
    return render(request, 'catalog/dmr_plot.html', context=context)

def lc50_plot(request):
    return
@login_required(login_url='/accounts/login/')
def dmp_plot (request):
    dmp = DMP.objects.all()
    chromosomes = request.GET.get('chromosome')

    if chromosomes:
        dmp = dmp.filter(chromosome = chromosomes) 
    
    project_data_dmp = [
        {
            'CHR' : x.chromosome,
            'BP' : x.position,
            'P': x.adjpvalue,
            'SNP':x.name,
            'ZSCORE':x.AveExpr,
            'EFFECTSIZE':x.t,
            'GENE':x.gene,
            'DISTANCE':x.B
        } for x in dmp
    ]
    df_dmp = pd.DataFrame(project_data_dmp['CHR'])
    df_ 
    df_dmp = pd.DataFrame(project_data_dmp)
    df_dmp['CHR'] = pd.to_numeric(df_dmp['CHR'])
    df_dmp['BP'] = pd.to_numeric(df_dmp['BP'])
    df_dmp['P'] = pd.to_numeric(df_dmp['P'])
    df_dmp['ZSCORE'] = pd.to_numeric(df_dmp['ZSCORE'])
    df_dmp['EFFECTSIZE'] = pd.to_numeric(df_dmp['EFFECTSIZE'])
    df_dmp['DISTANCE'] = pd.to_numeric(df_dmp['DISTANCE'])
    fig_dmp = dash_bio.ManhattanPlot(dataframe=df_dmp)
    man_plot_dmp = plot(fig_dmp,output_type="div", show_link=False, link_text="")
    context = {
        'plot_div': man_plot_dmp, 
        'form':DMPForm()  
    }
    return render(request, 'catalog/dmp_plot.html', context=context)

@login_required(login_url='/accounts/login/')
def plots(request):
    sample_list = sample_information.objects.all() 
    print(type(sample_list))
    form = RNAForm(request.POST,data_list=sample_list)
    gene_parquet = pd.read_parquet('static/geneAnno.parquet',engine = 'pyarrow')
    lc50 = pd.read_parquet('static/LC50_plots.parquet',engine = 'pyarrow')
    pid = pd.read_parquet('static/sampleid.parquet',engine = 'pyarrow')    
    df_plot = pd.DataFrame()
    genes = request.POST.get('gene')
    gene_types = request.POST.get('gene_type')
    chromosomes = request.POST.get('chromosome')
    drugs = request.POST.get('drug')
    subtypes = request.POST.get('subtype')
    
    groups = request.POST.get('groups')
    
    group = 'status'
    if genes:
        df = pq.read_pandas('static/RNAseqT.parquet', columns = [genes]).to_pandas()
        if drugs:
            lc50 = lc50[((lc50['drug'] == drugs ) )] 
        if subtypes:
            lc50 = lc50[((lc50['subtype'] == subtypes ) )] 
        if groups:
            group = groups
        df['pcgpids'] =list( pid.pcgpids)
        df_plot = pd.merge(df, lc50,how='inner',on='pcgpids')
        fig = px.violin(
            df_plot,x = group, y = genes,points="all",box = True, color = group, hover_name = 'subtype'
            ,title= drugs,height=1000
            )
        plots = plot(fig, output_type="div")
        context = {'plot_div': plots,'form': form}

    else:
        context = {'form': form}
    return render(request, 'catalog/plots.html',context)
@login_required(login_url='/accounts/login/')
def general(request):
    sample_list = sample_information.objects.all()
    form = FormForm(request.POST,data_list=sample_list)
    X =  request.POST.get('x_axis')
    Y =  request.POST.get('y_axis')
    groups = request.POST.get('groups')
    groups2 = request.POST.get('groups2')
    drugs = request.POST.get('drugs')
    pid = pd.read_parquet('static/sampleid.parquet',engine = 'pyarrow')
    lc50 = pd.read_parquet('static/lc50_plot_category.parquet',engine = 'pyarrow')
    if X: 
        if X in lc50.columns:
            if Y in lc50.columns:
                fig = px.density_heatmap(lc50,x = X,y = Y,marginal_x = 'histogram',marginal_y='histogram',height=800)
                plots = plot(fig, output_type="div",show_link = False,link_text="")
                context = {'plot_div': plots,'form': form}
            else:
                df = pq.read_pandas('static/RNAseqT.parquet', columns = [Y]).to_pandas()
                df['pcgpids'] =list( pid.pcgpids)
                df_plot = pd.merge(df, lc50,how='inner',on='pcgpids')
                fig = px.violin(df_plot,x = X, y = Y,box = True ,color = X,height=800)
                plots = plot(fig, output_type="div")
                context = {'plot_div': plots,'form': form}
        else:
            if Y in lc50.columns:
                df = pq.read_pandas('static/RNAseqT.parquet', columns = [X]).to_pandas()
                df['pcgpids'] =list( pid.pcgpids)
                df_plot = pd.merge(df, lc50,how='inner',on='pcgpids')
                fig = px.violin(df_plot,x = X, y = Y,box = True ,color = Y,height=800,)
                plots = plot(fig, output_type="div")
                context = {'plot_div': plots,'form': form}
            else:
                df = pq.read_pandas('static/RNAseqT.parquet', columns = [X,Y]).to_pandas()
                df['pcgpids'] =list( pid.pcgpids)
                lc50_drug = lc50[lc50['drug']==drugs]
                df_plot = pd.merge(df, lc50_drug,how='inner',on='pcgpids')
                if groups:
                    group = groups
                    if groups2:
                        fig = px.scatter(df_plot,x = X, y = Y,trendline="ols",color = group, facet_col=groups2  ,marginal_x ='box',marginal_y ='box',height=800)
                    else:
                        fig = px.scatter(df_plot,x = X, y = Y,trendline="ols",color = group ,marginal_x ='box',marginal_y ='box',height=800)
                else:
                    fig = px.scatter(df_plot,x = X, y = Y,trendline="ols",marginal_x ='box',marginal_y ='box',height=800)
                plots = plot(fig, output_type="div")
                context = {'plot_div': plots,'form': form}
    else:
        context = {'form':form}
    

    return render(request, 'catalog/general.html',context)
@login_required(login_url='/accounts/login/')
def dashtest(request):
    sample_list = sample_information.objects.all()
    form = FormForm2(request.POST,data_list=sample_list)
    X =  request.POST.get('x_axis') 
    Y =  request.POST.get('y_axis')
    pid = pd.read_parquet('static/pknumbers.parquet',engine = 'pyarrow')
    lc50 = pd.read_parquet('static/lc50_plot_category2.parquet',engine = 'pyarrow')
    cnv_del = pd.read_parquet('static/cnv_del.parquet',engine = 'pyarrow')
    groups = request.POST.get('groups')
    df_plot = pd.DataFrame()
    if X:
        if not X in lc50.columns:
            if not Y in lc50.columns:
                X = X.upper()
                Y = Y.upper()
                df = pq.read_pandas('static/RNAseqT.parquet', columns = [X,Y]).to_pandas()
                df['pharmgkbnumber'] =list(pid.pharmgkbnumber )
                df_plot = pd.merge(df, lc50,how='inner',on='pharmgkbnumber')
                df_plot = df_plot.drop(columns=['id','PercentBlastsDay4','pcgpids','wentao_subtype','subtype_pre'])
            else:
                X = X.upper()
                df = pq.read_pandas('static/RNAseqT.parquet', columns = [X]).to_pandas()
                df['pharmgkbnumber'] =list( pid.pharmgkbnumber)
                df_plot = pd.merge(df, lc50,how='inner',on='pharmgkbnumber')
                df_plot = df_plot.drop(columns=['id','PercentBlastsDay4','pcgpids','wentao_subtype','subtype_pre'])
        else:
            if  not Y in lc50.columns:
                Y = Y.upper()
                df = pq.read_pandas('static/RNAseqT.parquet', columns = [Y]).to_pandas()
                df['pharmgkbnumber'] =list( pid.pharmgkbnumber)
                df_plot = pd.merge(df, lc50,how='inner',on='pharmgkbnumber')
                df_plot = df_plot.drop(columns=['id','PercentBlastsDay4','pcgpids','wentao_subtype','subtype_pre'])
            else:
                df_plot = lc50
                df_plot = df_plot.drop(columns=['id','PercentBlastsDay4','pcgpids','wentao_subtype','subtype_pre'])
    app = DjangoDash('app1',external_stylesheets=[dbc.themes.BOOTSTRAP])
    controls =  html.Div([
            html.Div([
            html.Div(children=[
                dcc.Checklist(id='trendline',options=['Trendlines'],value='Trendlines')],style={'width': '25%', 'display': 'inline-block'}),
            html.Div(children=[
                dcc.Checklist(id='boxplot',options=['Boxplots'],value='Boxplots')],style={'width': '25%', 'display': 'inline-block'})
                ]),
            html.Div([
                html.Div(children=[
                html.Label('Drug'),
                dcc.Dropdown(lc50['drug'].unique(),list(lc50['drug'].unique()), multi=True,id='drug-filter'),
                html.Label('Age'),
                dcc.Dropdown(lc50['age_group'].unique(),['AYA','Ped','Older','Ad','Missing'], multi=True,id='age-filter'),
                html.Label('Lineage'),
                dcc.Dropdown(lc50['lineage'].unique(),['T','B','U','Mixed'], multi=True,id='lineage-filter',),
                html.Label('Status'),
                dcc.Dropdown(lc50['status'].unique(),['sensitive','resistant'],multi=True,id='status-filter'),
                html.Label('Disease Status'),
                dcc.Dropdown(lc50['diseaseStatus'].unique(),['PRIMARY','RELAPSE','REFRACTORY'],multi=True,id='disease-filter',),
                html.Label('Subtype'),
                dcc.Dropdown(lc50['subtype'].unique(),list(lc50['subtype'].unique()), multi=True,id='subtype-filter',maxHeight=100,),
                ],style={'width': '100%', 'display': 'inline-block'} )
                ]),], style={'display': 'block'}, id='filters-container')            

    app.layout = html.Div([
        "Filters",
        dcc.Dropdown(
        id='filters',
        options=[
            {'label': 'Show', 'value': 'on'},
            {'label': 'Hide', 'value': 'off'}
        ],
        clearable=False,
        multi=False,
        value='off',
        style={'width': '80px'}
    ),
        dbc.Row([
                 dbc.Col(dcc.Graph(id="indicator-graphic",config={"displaylogo": False,'toImageButtonOptions': {
                                                                                       'format': 'svg', 'filename': 'custom_image',
                                                                                       'height': 700,'width': 1000,'scale': 1 }}), md=8),
                dbc.Col(controls, md=4), 
                ]),
    dash_table.DataTable(
        id='datatable-interactivity',
        columns=[
            {"name": i, "id": i, "deletable": False, "selectable": True, "filter_options":{"case":"sensitive"},
            "hideable":True} for i in df_plot.columns
        ],
        data=df_plot.to_dict('records'),
        editable=True,
        filter_action="native",
        sort_action="native",
        sort_mode="multi",
        row_deletable=True,
        page_action="native",
        page_current= 0,
        page_size= 10,
        hidden_columns=['diseaseStatus','assay_type','rsquared'],
        style_data={
        'width':'10%'
    },
    ),
    html.Div(id='datatable-interactivity-container')
    ])
    @app.callback(
    Output(component_id='filters-container', component_property='style'),
    [Input(component_id='filters', component_property='value')]
    )
    def show_hide (filters):
        if filters == 'on':
            return {'display': 'block'}
        if filters  == 'off':
            return {'display': 'none'}

    @app.callback(Output('indicator-graphic', 'figure'), 
                  Input('datatable-interactivity', "derived_virtual_data"),
                  Input('drug-filter', 'value'),
                  Input('age-filter', 'value'),
                  Input('subtype-filter', 'value'),
                  Input('status-filter', 'value'),
                  Input('lineage-filter', 'value'),
                  Input('disease-filter', 'value'),
                  Input('trendline','value'),
                  Input('boxplot','value'),
                  )
    def update_graph(rows,drug_filter,age_filter,subtype_filter,status_filter,lineage_filter,diseasestatus_filter,trendline,boxplot):
        dfff =  pd.DataFrame(rows)
        dff = dfff[dfff['pharmgkbnumber'.isin(df_plot)]]
        dff = dff[dff['drug'].isin(drug_filter)]
        dff = dff[dff['subtype'].isin(subtype_filter)]
        dff = dff[dff['age_group'].isin(age_filter)]   
        dff = dff[dff['status'].isin(status_filter)] 
        dff = dff[dff['lineage'].isin(lineage_filter)]
        dff = dff[dff['diseaseStatus'].isin( diseasestatus_filter)]
        if X:
            if X in lc50.columns:
                if Y in lc50.columns:
                    fig = px.histogram(dff, x=X, color = Y,height = 800)
                else:
                    fig = px.violin(dff,x = X, y = Y,points = 'all',box = True ,color = X,height=800) 
            else:
                if Y in lc50.columns:
                    fig = px.violin(dff,x = X, y = Y,points = 'all',box = True ,color = Y,height=800,)
                else:
                    if groups:
                        if 'Trendlines' in trendline:
                            if 'Boxplots' in boxplot:
                                fig = px.scatter(dff, x=X, y=Y,trendline="ols",color = groups,marginal_x ='box',marginal_y ='box',height = 800)
                            else:
                                fig = px.scatter(dff, x=X, y=Y,trendline="ols",color = groups,height = 800)
                        else:
                            if 'Boxplots' in boxplot:
                                fig = px.scatter(dff, x=X, y=Y,color = groups,marginal_x ='box',marginal_y ='box',height = 800)
                            else:
                                fig = px.scatter(dff, x=X, y=Y,color = groups,height = 800)
                    else:
                        if 'Trendlines' in trendline:  
                            if 'Boxplots' in boxplot:
                                fig = px.scatter(dff, x=X, y=Y,trendline="ols" ,marginal_x ='box',marginal_y ='box',height = 800)
                            else:
                                fig = px.scatter(dff, x=X, y=Y,trendline="ols",height = 800)
                        else:
                            if 'Boxplots' in boxplot:
                                fig = px.scatter(dff, x=X, y=Y ,marginal_x ='box',marginal_y ='box',height = 800)
                            else:
                                fig = px.scatter(dff, x=X, y=Y,height = 800)
        fig.update_layout(
                      width = 800,
                      height = 600,
                      autosize=True,
                      template="plotly_white",
                      )
        return fig

    context = {'form': form}
    return render(request, 'catalog/app.html',context)
@login_required(login_url='/accounts/login/')
def sample_overview(request):
    df = pd.read_csv('static/Contrast_institution.csv').set_index('pharmgkbnumber')
    df = df.drop(columns=['Unnamed: 0','Unnamed: 0.1', 'Unnamed: 0.2'])
    columns = list(df.columns.values)
    columns.remove('institution')
    rows = list(df['institution'].unique())
    app = DjangoDash('app_overview',external_stylesheets=[dbc.themes.BOOTSTRAP])
    controls =  html.Div([
                html.Div([ 
                html.Div(children=[
                html.Label('Institution'),
                dcc.Dropdown(rows,['SJCRH','Missing'], multi=True,id='institution-filter'),
                html.Label('Drug'),
                dcc.Dropdown(columns,columns, multi=True,id='drug-filter'),
                ],style={'width': '50%', 'display': 'inline-block'} )
                ]),])
    app.layout = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(controls, md=4),
                dbc.Col(dcc.Graph(id="indicator-graphic",config={"displaylogo": False,}), md=6),
            ],
        ),
    ],
    )
    @app.callback(Output('indicator-graphic', 'figure'),
                  Input('institution-filter','value'),
                  Input('drug-filter','value'),)
    def update_clustergram(institution,drug):
        dff = df[df['institution'].isin(institution)]
        
        
        
        fig = px.imshow( 
            dff[drug].T , aspect = 'auto', width =800, height = 800,origin = 'lower',
            )
        return fig
    context = {}
    return render(request, 'catalog/drugoverview.html',context)
@login_required(login_url='/accounts/login/')
def overall (request):
    pid = pd.read_parquet('static/pknumbers.parquet',engine = 'pyarrow')
    lc50 = pd.read_csv('static/Web_LC50_newlc50.csv')
    ###lc50 = lc50.drop(columns = ['Unnamed: 0'])
    patient = pd.read_csv('static/Web_Patients.csv')
  
    df_plot = patient.drop(columns = ['Unnamed: 0'])
    cnv = pd.read_parquet('static/cnv_del.parquet',engine = 'pyarrow')
    snv = pd.read_parquet('static/Web_snv_fillna.parquet',engine = 'pyarrow')
    ###patient_lc50 = pd.merge(patient, lc50,how='inner',on='pharmgkbnumber')
    d =  pd.read_parquet('static/Web_dtype.parquet',engine = 'pyarrow')
    dropdown = pd.read_parquet('static/Web_dropdown_list.parquet',engine = 'pyarrow')
    dropdown = list(dropdown.list)
    
    app = DjangoDash('app_overall',external_stylesheets=[dbc.themes.BOOTSTRAP])
    col = [
    {"label": i , "value": i } for i in dropdown
    ]
    plot_options = df_plot.columns
    controls =  html.Div([
            html.Div([
            html.Div(children=[
                dcc.Checklist(id='trendline',options=['Trendlines'],value='Trendlines'),
                'X axis',
                dcc.Dropdown(plot_options,value = 'lineage',id='x-axis',),],style={'width': '25%', 'display': 'inline-block'}),
            html.Div(children=[
                dcc.Checklist(id='boxplot',options=['Boxplots'],value='Boxplots'),
                'Y axis',
                dcc.Dropdown(plot_options,value = 'subtype_clean',id='y-axis'),],style={'width': '25%', 'display': 'inline-block'}),
            html.Div(children=[
                dcc.Checklist(id='Reverse',options=['Reverse'],value=''),
                'groups',
                dcc.Dropdown(plot_options,value = 'lineage',id='groups')],style={'width': '25%', 'display': 'inline-block'}),
                ]),
                ], style={'display': 'block'}, id='filters-container')
    table_control = html.Div([ 
            html.Div (children=[html.Button('Add Column', id='editing-columns-button', n_clicks=0)],style={'width': '92%', 'display': 'inline-block'}),
            html.Div (children=[html.Button('Reset Table', id='reset-button', n_clicks=0)],style={ 'display': 'inline-block'})
            ]) 
    data_edit = html.Div([ 
    html.Div (children=[dcc.Input(id='column-input',placeholder='Enter the column',debounce=True)],style={'width': '30%', 'display': 'inline-block'} ),
    html.Div (children=[dcc.Input(id='from-input',placeholder='Enter the old value',debounce=True)],style={'width': '30%', 'display': 'inline-block'} ),
    html.Div (children=[dcc.Input(id='to-input',placeholder='Enter the new value',debounce=True)],style={'width': '30%', 'display': 'inline-block'} ),
    ])
    app.layout = html.Div([
        dcc.Input(id='filter-query-input', placeholder='Enter filter query     "Example: {TP53}>8 and {LCK}<12 and {drug} contains DEX and..."',debounce=True,style={'width':'1000px'} ),
        html.Div([dcc.Dropdown(multi=False,placeholder='Enter data columns: ',id='editing-columns-name',style={'width':'50%'})]),
        dbc.Row([dbc.Col(table_control)]),
        dash_table.DataTable(
          id='datatable-interactivity',
          columns=[
            {"name": i, "id": i, "deletable": False, "selectable": False,"hideable":True} for i in df_plot.columns
            ],
          data=df_plot.to_dict('records'),
          editable=True,
          filter_action="native",
          cell_selectable  = True,
          sort_action="native",
          sort_mode="multi",
          row_deletable=True,
          page_action="native",
          page_current= 0,
          page_size= 10,
          hidden_columns=['protocol','accession','assay_start_date','sample_date','date_of_birth','diseaseStatus',
                          'test_mnemonic','storage','blasts_qc','PercentBlasts','PercentBlastsPostFicoll',
                          'PercentBlastsDay4','assay_type','age_group'],
          export_format='xlsx',export_headers='display',merge_duplicate_headers=True,
          style_table={'overflowX': 'auto'},
         style_cell={'textOverflow': 'ellipsis'},style_data_conditional=[{'if': {'row_index': 'odd'},'backgroundColor': 'rgb(220, 220, 220)'}], 
       style_header={'backgroundColor': 'white','color': 'black','fontWeight': 'bold'}
    ,),
    html.Div(id='datatable-interactivity-container'),
    dbc.Row([dbc.Col(data_edit)]),
    dbc.Row([dbc.Col(controls)]),
    dcc.Graph(id="indicator-graphic",config={"displaylogo": False,'toImageButtonOptions': {
                                                                                       'format': 'svg', 'filename': 'custom_image',
                                                                                       'height': 700,'width': 1000,'scale': 1 }}),
    ])

    @app.callback(
    Output("editing-columns-name", "options"),
    Input("editing-columns-name", "search_value")
    )
    def update_options(search_value):
        if not search_value:
            raise PreventUpdate
        elif len(search_value)<3:
            raise PreventUpdate 
        return [o for o in col if search_value.upper() in o["label"]]
    
    @app.callback(
    [dash.dependencies.Output("datatable-interactivity", 'data'), dash.dependencies.Output("datatable-interactivity", "columns")],
    Input('editing-columns-button', 'n_clicks'),
    Input('column-input', "value"),
    Input('from-input', "value"),
    Input('to-input', "value"),
    State('editing-columns-name', 'value'),
    State('datatable-interactivity', "derived_virtual_data"),
    State('datatable-interactivity', 'columns')
    )

    def update_columns(n_clicks,Column,From,To,value,data,columns):        
        d_target = d[[value]]
        df = pd.DataFrame(data)
        if n_clicks >0:
            if value == 'LC50':
                df = pd.merge(df,lc50,how='inner', on ='pharmgkbnumber')
            elif 'RNA' in value:
                gene = value.split("_",1)[0]
                rna = pq.read_pandas('static/RNAseqT.parquet', columns = [gene]).to_pandas()
                rna['pharmgkbnumber'] =list(pid.pharmgkbnumber)
                df = pd.merge(df, rna,how='inner',on='pharmgkbnumber')
            elif 'del' in value:           
                cnv_del = cnv[[value,'timepoint_del','pharmgkbnumber']]
                df = pd.merge(df,cnv_del,how='left', on ='pharmgkbnumber')
            elif 'SNV' in value:
                snv_target = snv[['pharmgkbnumber','timepoint',value]]
                df = pd.merge(df,snv_target,how='left', on ='pharmgkbnumber')
            elif 'EDIT' in value:
                if To != '': 
                    df.loc[df[Column]== From ,Column] = To
            else:
                df = df
          
            columns = [
            {"name": i, "id": i, "deletable": False, "selectable": True,"hideable":True} for i in df.columns
            ]
            data = df.to_dict('records')
            
        return data,columns

    @app.callback(
    Output("x-axis", "options"),
    Input('datatable-interactivity', 'derived_virtual_data'),
    )
    def update__plot_options(data):
        dfff = pd.DataFrame(data)
        plot_options = list(dfff.columns)
        return plot_options
    @app.callback(
    Output("y-axis", "options"),
    Input('datatable-interactivity', 'derived_virtual_data'),
    )
    def update__plot_options(data):
        dfff = pd.DataFrame(data)
        plot_options = list(dfff.columns)
        return plot_options
    @app.callback(
    Output("groups", "options"),
    Input('datatable-interactivity', 'derived_virtual_data'),
    )
    def update__plot_options(data):
        dfff = pd.DataFrame(data)
        plot_options = list(dfff.columns)
        return plot_options
    @app.callback(
    Output('datatable-interactivity', 'filter_query'),
    Input('filter-query-input', 'value')
    )
    def write_query(query):
        if query is None:
            return ''
        return query

    @app.callback(
    Output('datatable-interactivity-container', "children"),
    Input('datatable-interactivity', "derived_filter_query_structure"),
    )
        
    @app.callback(
    Output('indicator-graphic', "figure"),
    [Input('x-axis', 'value'),Input('y-axis', 'value'),Input('groups','value'),Input('trendline','value'),Input('boxplot','value'),Input('Reverse','value')],
    Input('datatable-interactivity', "derived_virtual_data"),
    )
    def update_graphs(X,Y,groups,trendline,boxplot,reverse,data):
        dff = pd.DataFrame(data)
        if 'Reverse' in reverse:
            XX = X
            X = Y
            Y = XX
        if X:
            if Y:
                dff = dff.dropna(subset=[X,Y,groups]) 
                if is_string_dtype(dff[X]):
                    if is_string_dtype(dff[Y]):
                        fig = px.histogram(dff, x=X, color = Y,height = 800)
                    else:
                        fig = px.violin(dff,x = X, y = Y,points = 'all',color = X,height=800) 
                else:
                    if is_string_dtype(dff[Y]):
                        fig = px.violin(dff,x = X, y = Y,color = Y,points = 'all',height=800)
                    else:
                        if groups:
                            if 'Trendlines' in trendline:
                                if 'Boxplots' in boxplot:
                                    fig = px.scatter(dff, x=X, y=Y,trendline="ols",color = groups,marginal_x ='box',marginal_y ='box',height = 800)
                                else:
                                    fig = px.scatter(dff, x=X, y=Y,trendline="ols",color = groups,height = 800)
                            else:
                                if 'Boxplots' in boxplot:
                                    fig = px.scatter(dff, x=X, y=Y,color = groups,marginal_x ='box',marginal_y ='box',height = 800)
                                else:
                                    fig = px.scatter(dff, x=X, y=Y,color = groups,height = 800)
                        else:
                            if 'Trendlines' in trendline:  
                                if 'Boxplots' in boxplot:
                                    fig = px.scatter(dff, x=X, y=Y,trendline="ols" ,marginal_x ='box',marginal_y ='box',height = 800)
                                else:
                                    fig = px.scatter(dff, x=X, y=Y,trendline="ols",height = 800)
                            else:
                                if 'Boxplots' in boxplot:
                                    fig = px.scatter(dff, x=X, y=Y ,marginal_x ='box',marginal_y ='box',height = 800)
                                else:
                                    fig = px.scatter(dff, x=X, y=Y,height = 800) 
                fig.update_layout(
                      width = 1000,
                      height = 750,
                      autosize=True,
                      template="plotly_white",
                      )
        return fig
    @app.callback(
    Output("datatable-interactivity", "derived_virtual_data"),
    Input('reset-button', 'n_clicks'),
    )
    def reset_table(reset):
        if reset > 0:
            data = df_plot.to_dict('records')
        return data
    
    context = {}
    return render(request, 'catalog/overall.html',context)
@login_required(login_url='/accounts/login/')
def snv (request):
    snv = pd.read_parquet('static/Web_snv_fillna.parquet',engine = 'pyarrow')
    app = DjangoDash('app_snv',external_stylesheets=[dbc.themes.BOOTSTRAP])
    app.layout = html.Div([
    dcc.Input(id='filter-query-input', debounce=True, placeholder='Enter filter query     "Example: {TP53}>8 and {LCK}<12 and {drug} contains DEX and..."',style={'width':'1000px'} ),
    dash_table.DataTable(
          id='datatable-interactivity',
          columns=[
            {"name": i, "id": i, "deletable": False, "selectable": True,"hideable":True} for i in snv.columns
            ],
          data=snv.to_dict('records'),
          editable=True,
          filter_action="native",
          sort_action="native",
          sort_mode="multi",
          row_deletable=True,
          page_action="native",
          page_current= 0,
          page_size= 30,
          hidden_columns=snv.columns[10:],
          export_format='xlsx',export_headers='display',merge_duplicate_headers=True,
          style_data={
          'width':'10%'
          },
          ),
    html.Div(id='datatable-interactivity-container'),
    ])

 
    @app.callback(
    Output('datatable-interactivity', 'filter_query'),
    Input('filter-query-input', 'value')
    )
    def write_query(query):
        if query is None:
            return ''
        return query
    @app.callback(
    Output('datatable-interactivity-container', "children"),
    Input('datatable-interactivity', "derived_filter_query_structure"),
    )
    def temp():
        return
    context = {}
    return render(request, 'catalog/snv.html',context)
@login_required(login_url='/accounts/login/')
def cnv (request):
    cnv = pd.read_parquet('static/cnv_del.parquet',engine = 'pyarrow')
    app = DjangoDash('app_cnv',external_stylesheets=[dbc.themes.BOOTSTRAP])
    app.layout = html.Div([
    dcc.Input(id='filter-query-input', debounce=True, placeholder='Enter filter query     "Example: {TP53}>8 and {LCK}<12 and {drug} contains DEX and..."',style={'width':'1000px'} ),
    dash_table.DataTable(
          id='datatable-interactivity',
          columns=[
            {"name": i, "id": i, "deletable": False, "selectable": True,"hideable":True} for i in cnv.columns
            ],
          data=cnv.to_dict('records'),
          editable=True,
          filter_action="native",
          sort_action="native",
          sort_mode="multi",
          row_deletable=True,
          page_action="native",
          page_current= 0,
          page_size= 30,
          hidden_columns=cnv.columns[9:273],
          export_format='xlsx',export_headers='display',merge_duplicate_headers=True,
          style_data={
          'width':'10%'
          },
          ),
    html.Div(id='datatable-interactivity-container'),
    ])


    @app.callback(
    Output('datatable-interactivity', 'filter_query'),
    Input('filter-query-input', 'value')
    )
    def write_query(query):
        if query is None:
            return ''
        return query
    @app.callback(
    Output('datatable-interactivity-container', "children"),
    Input('datatable-interactivity', "derived_filter_query_structure"),
    )
    def temp():
        return
    context = {}
    return render(request, 'catalog/cnv.html',context)


from django.views import generic
from django.contrib.auth.mixins import LoginRequiredMixin
from django_tables2 import SingleTableView
from .tables import SampleTable, ProbeTable, DMPTable,DMRTable,LC50Table,DiscoveryTable,ScreenTable
from django_filters.views import FilterView
from django_tables2.views import SingleTableMixin
from .filters import *
from django_tables2.export.views import ExportMixin
import django_tables2 as tables
from django_tables2.export.export import TableExport




class SampleListView(LoginRequiredMixin, FilterView,ExportMixin, tables.SingleTableView):
    model = sampleid
    paginate_by = 20
    table_class = SampleTable
    filterset_class = SampleFilter

class SampleDetailView(generic.DetailView):
    model = sampleid
class ProbeListView (LoginRequiredMixin, FilterView,ExportMixin, tables.SingleTableView):
    model = MethylationProbe
    paginate_by = 20
    table_class = ProbeTable
    filterset_class = ProbeFilter
class ProbeDetailView(generic.DetailView):
    model = MethylationProbe
class DMPListView (LoginRequiredMixin, FilterView,ExportMixin, tables.SingleTableView):
    model = DMP
    paginate_by = 20
    table_class = DMPTable
    filterset_class =DMPFilter
class DMPDetailView(generic.DetailView):
    model = DMP
class DMRListView (LoginRequiredMixin, FilterView,ExportMixin, tables.SingleTableView):
    model = DMR
    paginate_by = 20
    table_class = DMRTable
    filterset_class = DMRFilter
class DMRDetailView(generic.DetailView):
    model = DMR
class Lc50ListView(LoginRequiredMixin, FilterView,ExportMixin, tables.SingleTableView):
    model = lc50
    paginate_by = 20
    table_class = LC50Table
    filterset_class = LC50Filter
class DiscoveryListView(LoginRequiredMixin, FilterView,ExportMixin, tables.SingleTableView):
    model = discovery
    paginate_by = 20
    table_class = DiscoveryTable
    filterset_class = DiscoveryFilter
class ScreenListView(LoginRequiredMixin, FilterView,ExportMixin, tables.SingleTableView):
    model = screen
    paginate_by = 20
    table_class = ScreenTable
    filterset_class = DrugFilter
class TestAutocomplete(autocomplete.Select2QuerySetView):
    def get_queryset(self):
        test = sample_information.objects.all()
        if self.q:
            test = test.filter(name__istartswith=self.q)

        return test







# Create your views here.
