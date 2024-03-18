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
import base64
import datetime
import io
import cv2
import numpy as np
import scipy.stats
from scipy import stats
import pdfkit
import dash_html_components as html
from pylatex import *
import plotly.io as pio
from pylatex.utils import bold, NoEscape
import anndata
import scanpy as sc
from scipy.sparse import csr_matrix, save_npz
from scipy.sparse import load_npz
import matplotlib.pyplot as plt

@login_required(login_url='/accounts/login/')
def index(request):
    


    
    context = {}
    return render(request, 'index.html', context=context)
@login_required(login_url='/accounts/login/')
def sample (request):     
    patient = pd.read_csv('static/Web_Patients.csv')
    patient = patient.drop(columns = ['Unnamed: 0'])
    app = DjangoDash('app_sample',external_stylesheets=[dbc.themes.BOOTSTRAP])
    app.layout = html.Div([
    dcc.Input(id='filter-query-input', debounce=True, placeholder='Enter filter query     "Example: {TP53}>8 and {LCK}<12 and {drug} contains DEX and..."',style={'width':'1000px'} ),
    dash_table.DataTable(
          id='datatable-interactivity',
          columns=[
            {"name": i, "id": i, "deletable": False, "selectable": True,"hideable":True} for i in patient.columns
            ],
          data=patient.to_dict('records'),
          editable=True,
          filter_action="native",
          sort_action="native",
          sort_mode="multi",
          row_deletable=True,
          page_action="native",
          page_current= 0,
          page_size= 15,
          hidden_columns=[],
          export_format='xlsx',export_headers='display',merge_duplicate_headers=True,
          style_table={'overflowX': 'auto'},
         style_cell={'textOverflow': 'ellipsis','font-family':'Helvetica'},style_data_conditional=[{'if': {'row_index': 'odd'},'backgroundColor': 'rgb(220, 220, 220)'}],
       style_header={'backgroundColor': 'white','color': 'black','fontWeight': 'bold'}
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
    return render(request, 'catalog/sample.html', context=context)
@login_required(login_url='/accounts/login/')
def drug(request):
    lc50 = pd.read_csv('static/Web_Final_LC50.csv')
    app = DjangoDash('app_lc50',external_stylesheets=[dbc.themes.BOOTSTRAP])
    app.layout = html.Div([
    dcc.Input(id='filter-query-input', debounce=True, placeholder='Enter filter query     "Example: {TP53}>8 and {LCK}<12 and {drug} contains DEX and..."',style={'width':'1000px'} ),
    dash_table.DataTable(
          id='datatable-interactivity',
          columns=[
            {"name": i, "id": i, "deletable": False, "selectable": True,"hideable":True} for i in lc50.columns
            ],
          data=lc50.to_dict('records'),
          editable=True,
          filter_action="native",
          sort_action="native",
          sort_mode="multi",
          row_deletable=True,
          page_action="native",
          page_current= 0,
          page_size= 15,
          hidden_columns=[],
          export_format='xlsx',export_headers='display',merge_duplicate_headers=True,
          style_table={'overflowX': 'auto'},
         style_cell={'textOverflow': 'ellipsis','font-family':'Helvetica'},style_data_conditional=[{'if': {'row_index': 'odd'},'backgroundColor': 'rgb(220, 220, 220)'}],
       style_header={'backgroundColor': 'white','color': 'black','fontWeight': 'bold'}
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
    return render(request, 'catalog/drug.html', context)

@login_required(login_url='/accounts/login/')
def discover(request):
    dis = pd.read_csv('static/Web_Discovery.csv')
    app = DjangoDash('app_discovery',external_stylesheets=[dbc.themes.BOOTSTRAP]) 
    app.layout = html.Div([
    dcc.Input(id='filter-query-input', debounce=True, placeholder='Enter filter query     "Example: {TP53}>8 and {LCK}<12 and {drug} contains DEX and..."',style={'width':'1000px'} ),
    dash_table.DataTable(
          id='datatable-interactivity',
          columns=[
            {"name": i, "id": i, "deletable": False, "selectable": True,"hideable":True} for i in dis.columns
            ],
          data=dis.to_dict('records'),
          editable=True,
          filter_action="native",
          sort_action="native",
          sort_mode="multi",
          row_deletable=True,
          page_action="native",
          page_current= 0,
          page_size= 15,
          hidden_columns=dis.columns[6:],
          export_format='xlsx',export_headers='display',merge_duplicate_headers=True,
          style_table={'overflowX': 'auto'},
         style_cell={'textOverflow': 'ellipsis','font-family':'Helvetica'},style_data_conditional=[{'if': {'row_index': 'odd'},'backgroundColor': 'rgb(220, 220, 220)'}],
       style_header={'backgroundColor': 'white','color': 'black','fontWeight': 'bold'}
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
    return render(request, 'catalog/discover.html', context)
@login_required(login_url='/accounts/login/')
def screening (request):
    screen = pd.read_parquet('static/Web_Screening.parquet')
    app = DjangoDash('app_screen',external_stylesheets=[dbc.themes.BOOTSTRAP])
    app.layout = html.Div([
    dcc.Input(id='filter-query-input', debounce=True, placeholder='Enter filter query     "Example: {TP53}>8 and {LCK}<12 and {drug} contains DEX and..."',style={'width':'1000px'} ),
    dash_table.DataTable(
          id='datatable-interactivity',
          columns=[
            {"name": i, "id": i, "deletable": False, "selectable": True,"hideable":True} for i in screen.columns
            ],
          data=screen.to_dict('records'),
          editable=True,
          filter_action="native",
          sort_action="native",
          sort_mode="multi",
          row_deletable=True,
          page_action="native",
          page_current= 0,
          page_size= 15,
          hidden_columns=[],
          export_format='xlsx',export_headers='display',merge_duplicate_headers=True,
          style_table={'overflowX': 'auto'},
         style_cell={'textOverflow': 'ellipsis','font-family':'Helvetica'},style_data_conditional=[{'if': {'row_index': 'odd'},'backgroundColor': 'rgb(220, 220, 220)'}],
       style_header={'backgroundColor': 'white','color': 'black','fontWeight': 'bold'}
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
    return render(request, 'catalog/screening.html',context)

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
def dmp_plot(request):
    return


def lc50_plot(request):
    return
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
    df_plot = pd.read_csv('static/Web_Patient_Final_with_SJ_LC50.csv',index_col=0)
    ###lc50 = lc50.drop(columns = ['Unnamed: 0'])
    cnv = pd.read_parquet('static/Web_cnv_del.parquet',engine = 'pyarrow')
    snv = pd.read_parquet('static/Web_snv_fillna.parquet',engine = 'pyarrow')
    ###df_plot = patient.drop(columns={'Unnamed: 0'})
    ###patient_lc50 = pd.merge(patient, lc50,how='inner',on='pharmgkbnumber')
    d =  pd.read_csv('static/Web_dtype.csv',index_col=0)
    dropdown = pd.read_parquet('static/Web_dropdown_list_nolc50.parquet',engine = 'pyarrow')
    dropdown = list(dropdown.list)
    app = DjangoDash('app_overall',external_stylesheets=[dbc.themes.BOOTSTRAP],add_bootstrap_links=True)
    app.css.append_css({ "external_url" : "/static/catalog/overall.css" })
    col = [
    {"label": i , "value": i } for i in dropdown
    ]
    plot_options = df_plot.columns
    plot_options= plot_options.append(pd.Index(['N/A']))
    controls =  html.Div([
            html.Div([
            html.Div(children=[
                'X axis',
                dcc.Dropdown(plot_options,value = 'Subtype',id='x-axis',),],style={'width': '25%', 'display': 'inline-block'}),
            html.Div(children=[
                'Y axis',
                dcc.Dropdown(plot_options,value = 'Subtype',id='y-axis'),],style={'width': '25%', 'display': 'inline-block'}),
            html.Div(children=[
                'group1',
                dcc.Dropdown(plot_options,value = 'N/A',id='group1')],style={'width': '25%', 'display': 'inline-block'}),
             html.Div(children=[
                'group2',
                dcc.Dropdown(plot_options,value = 'N/A',id='group2')],style={'width': '25%', 'display': 'inline-block'}),
                ]),
                ], style={'display': 'block'}, id='filters-container')
    plot_control =  html.Div([
            html.Div([
            html.Div(children=[
                dcc.Checklist(id='trendline',options=['Trendlines'],value=''),],style={'width': '7%', 'display': 'inline-block'}),
            html.Div(children=[
                dcc.Checklist(id='boxplot',options=['Boxplots'],value='Boxplots'),],style={'width': '7%', 'display': 'inline-block'}),
            html.Div(children=[
                dcc.Checklist(id='violin',options=['Violin'],value='Violin'),],style={'width': '7%', 'display': 'inline-block'}),
            html.Div(children=[
                dcc.Checklist(id='dot',options=['Dots'],value='Dots'),],style={'width': '7%', 'display': 'inline-block'}),
            html.Div(children=[
                dcc.Checklist(id='Reverse',options=['Reverse'],value=''),],style={'width': '7%', 'display': 'inline-block'}),
                ]),
                ], style={'display': 'block'}, id='check-container')
    table_control = html.Div([ 
            html.Div (children=[html.Button('Add Column', id='editing-columns-button', n_clicks=0)],style={'width': '92%', 'display': 'inline-block'}),
            ]) 
    app.layout = html.Div([
        dcc.Input(id='filter-query-input', placeholder='Enter filter query  "Example: {TP53}>8 and {LCK}<12 and {drug} contains DEX and {column_name}=... "',debounce=True,style={'width':'1000px'} ),
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
          filter_options = {'case':'insensitive'},
          cell_selectable  = True,
          sort_action="native",
          sort_mode="multi",
          row_deletable=True,
          page_action="native",
          page_current= 0,
          page_size= 10,
          hidden_columns=['institution','DiseaseStatus','AssayType','Original_LC50','Log10_LC50','Normalized_LC50','DataTimePoint','id_time'],
          export_format='xlsx',export_headers='display',merge_duplicate_headers=True,
          style_table={'overflowX': 'auto'},
         style_cell={'textOverflow': 'ellipsis','font-family':'Helvetica'},style_data_conditional=[{'if': {'row_index': 'odd'},'backgroundColor': 'rgb(220,220,220)'}], 
       style_header={'backgroundColor': 'white','color': 'black','fontWeight': 'bold'}
    ,),
    html.Div(id='datatable-interactivity-container'),
    dbc.Row([dbc.Col(plot_control)]),
    dbc.Row([dbc.Col(controls)]),
    dcc.Slider(750, 1350,marks={750:'Min',1000:'Default',1350:'Max'},id = 'slider-updatemode', value = 1000, updatemode='drag'),
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
    State('editing-columns-name', 'value'),
    State('datatable-interactivity', "derived_virtual_data"),
    State('datatable-interactivity', 'columns')
    )

    def update_columns(n_clicks,value,data,columns):        
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
                cnv_del = cnv[[value,'timepoint','pharmgkbnumber']]
                df = pd.merge(df,cnv_del,how='inner', on ='pharmgkbnumber')
            elif 'SNV' in value:
                snv_target = snv[['pharmgkbnumber','timepoint',value]]
                df = pd.merge(df,snv_target,how='inner', on ='pharmgkbnumber')
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
        plot_options.append('N/A')
        return plot_options
    @app.callback(
    Output("y-axis", "options"),
    Input('datatable-interactivity', 'derived_virtual_data'),
    )
    def update__plot_options(data):
        dfff = pd.DataFrame(data)
        plot_options = list(dfff.columns)
        plot_options.append('N/A')
        return plot_options
    @app.callback(
    Output("group1", "options"),
    Input('datatable-interactivity', 'derived_virtual_data'),
    )
    def update__plot_options(data):
        dfff = pd.DataFrame(data)
        plot_options = list(dfff.columns)
        plot_options.append('N/A')
        return plot_options
    @app.callback(
    Output("group2", "options"),
    Input('datatable-interactivity', 'derived_virtual_data'),
    )
    def update__plot_options(data):
        dfff = pd.DataFrame(data)
        plot_options = list(dfff.columns)
        plot_options.append('N/A')
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
    [Input('x-axis', 'value'),Input('y-axis', 'value'),Input('group1','value'),Input('group2','value'),Input('trendline','value'),Input('boxplot','value'),
    Input('violin','value'),Input('dot','value'),Input('Reverse','value'),Input('slider-updatemode','value') ],
    Input('datatable-interactivity', "derived_virtual_data"),
    )
    def update_graphs(X,Y,groups,group2,trendline,boxplot,violin,dot,reverse,slider,data):
        dff = pd.DataFrame(data)
        if 'Reverse' in reverse:
            XX = X
            X = Y
            Y = XX
        if X:
            if Y:
                if groups != 'N/A' and group2 != 'N/A':
                    dff = dff.dropna(subset=[X,Y,groups,group2])
                elif groups != 'N/A' and group2 == 'N/A':
                    dff = dff = dff.dropna(subset=[X,Y,groups])
                elif groups == 'N/A' and group2 != 'N/A':
                    dff = dff.dropna(subset=[X,Y,group2])
                else:
                    dff = dff.dropna(subset=[X,Y])
                if is_string_dtype(dff[X]):
                    if is_string_dtype(dff[Y]):
                        fig = px.histogram(dff, x=X, color = Y,text_auto=True,height = 800).update_xaxes(categoryorder='total descending')
                    else:
                        cat = [dff[Y][dff[X] == category] for category in dff[X].unique()]
                        statistic, p = stats.kruskal(*cat)
                        if 'Boxplots' in boxplot:
                            if 'Violin' in violin:
                                if 'Dots' in dot:
                                    if group2 != 'N/A':
                                        fig = px.violin(dff,x = X, y = Y,points = 'all',color = groups,box=True,title = f"P-value: {p}") if groups != 'N/A' else px.violin(dff,x = X, y = Y,points = 'all',box=True,title = f"P-value: {p}")
                                    else:
                                        fig = px.violin(dff,x = X, y = Y,points = 'all',color = groups,box=True,title = f"P-value: {p}") if groups != 'N/A' else px.violin(dff,x = X, y = Y,points = 'all',box=True,title = f"P-value: {p}")
                                else:
                                    if group2 != 'N/A':
                                        fig = px.violin(dff,x = X,y=Y,points='outliers',color=groups,box=True,title = f"P-value: {p}") if groups != 'N/A' else px.violin(dff,x = X, y = Y,points = 'outliers',box=True,title = f"P-value: {p}")
                                    else:
                                        fig = px.violin(dff,x = X,y=Y,points='outliers',color=groups,box=True,title = f"P-value: {p}") if groups != 'N/A' else px.violin(dff,x = X, y = Y,points = 'outliers',box=True,title = f"P-value: {p}")
                            else:
                                if 'Dots' in dot:
                                    if group2 != 'N/A':
                                        fig = px.box(dff,x = X, y = Y,points = 'all',color = groups,title = f"P-value: {p}") if groups != 'N/A' else px.box(dff,x = X, y = Y,points = 'all',title = f"P-value: {p}")
                                    else:
                                        fig = px.box(dff,x = X, y = Y,points = 'all',color = groups,title = f"P-value: {p}") if groups != 'N/A' else px.box(dff,x = X, y = Y,points = 'all',title = f"P-value: {p}")
                                else:
                                    if group2 != 'N/A':
                                        fig = px.box(dff,x = X, y = Y,points = 'outliers',color = groups,title = f"P-value: {p}") if groups != 'N/A' else px.box(dff,x = X, y = Y,points = 'outliers',title = f"P-value: {p}")
                                    else:
                                        fig = px.box(dff,x = X, y = Y,points = 'outliers',color = groups,title = f"P-value: {p}") if groups != 'N/A' else px.box(dff,x = X, y = Y,points = 'outliers',title = f"P-value: {p}")
                        else:
                            if 'Violin' in violin:
                                if 'Dots' in dot:
                                    if group2 != 'N/A':
                                        fig = px.violin(dff,x = X, y = Y,points = 'all',color = groups,title = f"P-value: {p}") if groups != 'N/A' else px.violin(dff,x = X, y = Y,points = 'all',title = f"P-value: {p}")
                                    else:
                                        fig = px.violin(dff,x = X, y = Y,points = 'all',color = groups,title = f"P-value: {p}") if groups != 'N/A' else px.violin(dff,x = X, y = Y,points = 'all',title = f"P-value: {p}")
                                else:
                                    if group2 != 'N/A':
                                        fig = px.violin(dff,x=X, y = Y,points = 'outliers',color = groups,title = f"P-value: {p}") if groups != 'N/A' else px.violin(dff,x = X, y = Y,points = 'outliers',title = f"P-value: {p}")
                                    else:
                                        fig = px.violin(dff,x=X, y = Y,points = 'outliers',color = groups,title = f"P-value: {p}") if groups != 'N/A' else px.violin(dff,x = X, y = Y,points = 'outliers',title = f"P-value: {p}")
                            else:
                                if 'Dots' in dot:
                                    if  group2 != 'N/A':
                                        fig = px.strip(dff,x = X, y = Y,color = groups,symbol=group2,title = f"P-value: {p}") if groups != 'N/A' else px.strip(dff,x = X, y = Y,symbol=group2,title = f"P-value: {p}")
                                    else:
                                        fig = px.strip(dff,x = X, y = Y,color = groups,title = f"P-value: {p}") if groups != 'N/A' else px.strip(dff,x = X, y = Y,title = f"P-value: {p}")
                else:
                    if is_string_dtype(dff[Y]):
                        cat = [dff[X][dff[Y] == category] for category in dff[Y].unique()]
                        statistic, p = stats.kruskal(*cat)
                        if 'Boxplots' in boxplot:
                            if 'Violin' in violin:
                                if 'Dots' in dot:
                                    if group2 != 'N/A':
                                        fig = px.violin(dff,x=X,y=Y,points='all',color=groups,box=True,title = f"P-value: {p}") if groups != 'N/A' else px.violin(dff,x = X, y = Y,points = 'all',box=True,title = f"P-value: {p}")
                                    else:
                                        fig = px.violin(dff,x=X,y=Y,points='all',color=groups,box=True,title = f"P-value: {p}") if groups != 'N/A' else px.violin(dff,x = X, y = Y,points = 'all',box=True,title = f"P-value: {p}")
                                else:
                                    if group2 != 'N/A':
                                        fig = px.violin(dff,x=X,y=Y,points='outliers',color=groups,box=True,title = f"P-value: {p}") if groups != 'N/A' else px.violin(dff,x=X,y=Y,points='outliers',box=True,title = f"P-value: {p}")
                                    else:
                                        fig = px.violin(dff,x=X,y=Y,points='outliers',color=groups,box=True,title = f"P-value: {p}") if groups != 'N/A' else px.violin(dff,x=X,y=Y,points='outliers',box=True,title = f"P-value: {p}")
                            else:
                                if 'Dots' in dot:
                                    if group2 != 'N/A':
                                        fig = px.box(dff,x = X, y = Y,points = 'all',color = groups,title = f"P-value: {p}") if groups != 'N/A' else px.box(dff,x = X, y = Y,points = 'all',title = f"P-value: {p}")
                                    else:
                                        fig = px.box(dff,x = X, y = Y,points = 'all',color = groups,title = f"P-value: {p}") if groups != 'N/A' else px.box(dff,x = X, y = Y,points = 'all',title = f"P-value: {p}")
                                else:
                                    if group2 != 'N/A':
                                        fig = px.box(dff,x = X, y = Y,points = 'outliers',color = groups,title = f"P-value: {p}") if groups != 'N/A' else px.box(dff,x = X, y = Y,points = 'outliers',title = f"P-value: {p}")
                                    else:
                                        fig = px.box(dff,x = X, y = Y,points = 'outliers',color = groups,title = f"P-value: {p}") if groups != 'N/A' else px.box(dff,x = X, y = Y,points = 'outliers',title = f"P-value: {p}")
                        else:
                            if 'Violin' in violin:
                                if 'Dots' in dot:
                                    if group2 != 'N/A':
                                        fig = px.violin(dff,x = X, y = Y,points = 'all',color = groups,title = f"P-value: {p}") if groups != 'N/A' else px.violin(dff,x = X, y = Y,points = 'all',title = f"P-value: {p}")
                                    else:
                                        fig = px.violin(dff,x = X, y = Y,points = 'all',color = groups,title = f"P-value: {p}") if groups != 'N/A' else px.violin(dff,x = X, y = Y,points = 'all',title = f"P-value: {p}")
                                else:
                                    if group2 != 'N/A':
                                        fig = px.violin(dff,x = X, y=Y,points = 'outliers',color = groups,title = f"P-value: {p}") if groups != 'N/A' else px.violin(dff,x = X, y = Y,points = 'outliers',title = f"P-value: {p}")
                                    else:
                                        fig = px.violin(dff,x = X, y = Y,points = 'outliers',color = groups,title = f"P-value: {p}") if groups != 'N/A' else px.violin(dff,x = X, y = Y,points = 'outliers',title = f"P-value: {p}")
                            else:
                                if 'Dots' in dot:
                                    if group2 != 'N/A':
                                        fig = px.strip(dff,x = X, y = Y,color = groups,symbol=group2,title = f"P-value: {p}") if groups != 'N/A' else px.strip(dff,x = X, y = Y,symbol=group2,title = f"P-value: {p}")
                                    else:
                                        fig = px.strip(dff,x = X, y = Y,color = groups,title = f"P-value: {p}") if groups != 'N/A' else px.strip(dff,x = X, y = Y,title = f"P-value: {p}")
                    else:
                        if groups != 'N/A':
                            if 'Trendlines' in trendline:
                                if 'Boxplots' in boxplot:
                                    if group2 != 'N/A':
                                        r, p = scipy.stats.pearsonr(dff[X], dff[Y])
                                        fig = px.scatter(dff, x=X, y=Y,trendline="ols",color = groups,symbol=group2,marginal_x ='box',marginal_y ='box',height = 800,
                                                          title = f"P-value: {p}")
                                    else:
                                        r, p = scipy.stats.pearsonr(dff[X], dff[Y])
                                        fig = px.scatter(dff, x=X, y=Y,trendline="ols",color = groups,marginal_x ='box',marginal_y ='box',height = 800,
                                                           title = f"P-value: {p}")
                                else:
                                    if group2 != 'N/A':
                                        r, p = scipy.stats.pearsonr(dff[X], dff[Y])
                                        fig = px.scatter(dff, x=X, y=Y,trendline="ols",color = groups,symbol=group2,height = 800,
                                                          title = f"P-value: {p}")
                                    else:
                                        r, p = scipy.stats.pearsonr(dff[X], dff[Y])
                                        fig = px.scatter(dff, x=X, y=Y,trendline="ols",color = groups,height = 800,
                                                         title = f"P-value: {p}")
                            else:
                                if 'Boxplots' in boxplot:
                                    if group2 != 'N/A':
                                        fig = px.scatter(dff, x=X, y=Y,color = groups,symbol=group2,marginal_x ='box',marginal_y ='box',height = 800)
                                    else:
                                        fig = px.scatter(dff, x=X, y=Y,color = groups,marginal_x ='box',marginal_y ='box',height = 800)
                                else:
                                    if group2 != 'N/A':
                                        fig = px.scatter(dff, x=X, y=Y,color = groups,symbol=group2,height = 800)
                                    else:
                                        fig = px.scatter(dff, x=X, y=Y,color = groups,height = 800)
                        else:
                            if 'Trendlines' in trendline:  
                                if 'Boxplots' in boxplot:
                                    if group2 != 'N/A':
                                        r, p = scipy.stats.pearsonr(dff[X], dff[Y])
                                        fig = px.scatter(dff, x=X, y=Y,trendline="ols",color=group2 ,marginal_x ='box',marginal_y ='box',height = 800,
                                                         title = f"P-value: {p}")
                                    else:
                                        r, p = scipy.stats.pearsonr(dff[X], dff[Y])
                                        fig = px.scatter(dff, x=X, y=Y,trendline="ols" ,marginal_x ='box',marginal_y ='box',height = 800,
                                                         title = f"P-value: {p}")
                                else:
                                    if group2 != 'N/A':
                                        r, p = scipy.stats.pearsonr(dff[X], dff[Y])
                                        fig = px.scatter(dff, x=X, y=Y,color=group2,trendline="ols",height = 800,
                                                         title = f"P-value: {p}")
                                    else:
                                        r, p = scipy.stats.pearsonr(dff[X], dff[Y])
                                        fig = px.scatter(dff, x=X, y=Y,trendline="ols",height = 800,
                                                         title = f"P-value: {p}")
                            else:
                                if 'Boxplots' in boxplot:
                                    if group2 != 'N/A':
                                        fig = px.scatter(dff, x=X, y=Y,color=group2 ,marginal_x ='box',marginal_y ='box',height = 800)
                                    else:
                                        fig = px.scatter(dff, x=X, y=Y ,marginal_x ='box',marginal_y ='box',height = 800)
                                else:
                                    if group2 != 'N/A':
                                        fig = px.scatter(dff, x=X, y=Y,color=group2,height = 800)
                                    else:
                                        fig = px.scatter(dff, x=X, y=Y,height = 800) 
                fig.update_layout(
                      width = slider,
                      height = slider-250,
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
          page_size= 15,
          hidden_columns=snv.columns[10:],
          export_format='xlsx',export_headers='display',merge_duplicate_headers=True,
          style_table={'overflowX': 'auto'},
         style_cell={'textOverflow': 'ellipsis','font-family':'Helvetica'},style_data_conditional=[{'if': {'row_index': 'odd'},'backgroundColor': 'rgb(220, 220, 220)'}],
       style_header={'backgroundColor': 'white','color': 'black','fontWeight': 'bold'}
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
    cnv = pd.read_parquet('static/Web_cnv_del.parquet',engine = 'pyarrow')
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
          page_size= 15,
          hidden_columns=cnv.columns[9:270],
          export_format='xlsx',export_headers='display',merge_duplicate_headers=True,
          style_table={'overflowX': 'auto'},
         style_cell={'textOverflow': 'ellipsis','font-family':'Helvetica'},style_data_conditional=[{'if': {'row_index': 'odd'},'backgroundColor': 'rgb(220, 220, 220)'}],
       style_header={'backgroundColor': 'white','color': 'black','fontWeight': 'bold'}
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

@login_required(login_url='/accounts/login/')
def seg(request):
    seg = pd.read_parquet('static/Web_seg.parquet',engine = 'pyarrow')
    app = DjangoDash('app_seg',external_stylesheets=[dbc.themes.BOOTSTRAP])
    app.layout = html.Div([
    dcc.Input(id='filter-query-input', debounce=True, placeholder='Enter filter query     "Example: {TP53}>8 and {LCK}<12 and {drug} contains DEX and..."',style={'width':'1000px'} ),
    dash_table.DataTable(
          id='datatable-interactivity',
          columns=[
            {"name": i, "id": i, "deletable": False, "selectable": True,"hideable":True} for i in seg.columns
            ],
          data=seg.to_dict('records'),
          editable=True,
          filter_action="native",
          sort_action="native",
          sort_mode="multi",
          row_deletable=True,
          page_action="native",
          page_current= 0,
          page_size= 15,
  
          export_format='xlsx',export_headers='display',merge_duplicate_headers=True,
          style_table={'overflowX': 'auto'},
         style_cell={'textOverflow': 'ellipsis','font-family':'Helvetica'},style_data_conditional=[{'if': {'row_index': 'odd'},'backgroundColor': 'rgb(220, 220, 220)'}],
       style_header={'backgroundColor': 'white','color': 'black','fontWeight': 'bold'}
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
    def temp(value):
        return value
    context = {}
    return render(request, 'catalog/segment.html',context)
@login_required(login_url='/accounts/login/')
def crispr(request):
    mp = pd.read_csv('static/CRISPR_ALL_REH.csv',index_col = 0)
    app = DjangoDash('app_crispr',external_stylesheets=[dbc.themes.BOOTSTRAP],add_bootstrap_links=True)
    app.layout = html.Div([
        dcc.Input(id='filter-query-input', placeholder='Enter filter query  "Example: {TP53}>8 and {LCK}<12 and {drug} contains DEX and {column_name}=... "',debounce=True,style={'width':'1000px'} ),
        html.Div([dcc.Dropdown(['6-MP','AraC','Daunorubicin','L-asparaginase','Maphosphamide','Methotrexate','Vincristine','Trametinib','Dasatinib','ALL'],'ALL',id='drug-name',style={'width':'50%'})]),
        dash_table.DataTable(
          id='datatable-interactivity',
          columns=[
            {"name": i, "id": i, "deletable": False, "selectable": False,"hideable":True} for i in mp.columns
            ],
          data= mp.to_dict('records'),
          editable=True,
          filter_action="native",
          filter_options = {'case':'insensitive'},
          cell_selectable  = True,
          sort_action="native",
          sort_mode="multi",
          row_deletable=True,
          page_action="native",
          page_current= 0,
          page_size= 20,
          export_format='xlsx',export_headers='display',merge_duplicate_headers=True,
          style_table={'overflowX': 'auto'},
         style_cell={'textOverflow': 'ellipsis','font-family':'Helvetica'},style_data_conditional=[{'if': {'row_index': 'odd'},'backgroundColor': 'rgb(220,220,220)'}], 
       style_header={'backgroundColor': 'white','color': 'black','fontWeight': 'bold'}
    ,),
    html.Div(id='datatable-interactivity-container'),
     ])
    @app.callback(
    [dash.dependencies.Output("datatable-interactivity", 'data'), dash.dependencies.Output("datatable-interactivity", "columns")],
    Input('drug-name', 'value'),
    State('datatable-interactivity', "derived_virtual_data"),
    State('datatable-interactivity', 'columns')
    )
    def update_columns(drug,data,columns):        
        if drug == '6-MP':
            df = pd.read_csv('static/CRISPR_6MP_REH.csv')
        elif drug == 'AraC':
            df = pd.read_csv('static/CRISPR_AraC_REH.csv') 
        elif drug == 'Daunorubicin':
            df = pd.read_csv('static/CRISPR_Daunorubicin_REH.csv')
        elif drug == 'L-asparaginase':
            df = pd.read_csv('static/CRISPR_L_asparaginase_REH.csv')
        elif drug == 'Maphosphamide':
            df = pd.read_csv('static/CRISPR_Maphosphamide_REH.csv')
        elif drug == 'Methotrexate':
            df = pd.read_csv('static/CRISPR_Methotrexate_REH2.csv',index_col = 0)
        elif drug == 'Trametinib':
            df = pd.read_parquet('static/Web_Screening.parquet')
            df = df[df['Drugs'].str.contains('Trametinib')]
        elif drug == 'Dasatinib':
            df = pd.read_parquet('static/Web_Screening.parquet')
            df = df[df['Drugs'].str.contains('Dasatinib')]
        elif drug == 'Vincristine':
            df = pd.read_csv('static/CRISPR_Vincristine_REH2.csv',index_col = 0)
        else:
            df = pd.read_csv('static/CRISPR_ALL_REH.csv',index_col = 0)

        columns = [
            {"name": i, "id": i, "deletable": False, "selectable": True,"hideable":True} for i in df.columns
            ]
        data = df.to_dict('records')
        return data,columns
    context = {}
    return render(request, 'catalog/crispr.html',context)

@login_required(login_url='/accounts/login/')
def image(request): 
    app = DjangoDash('app_image',external_stylesheets=[dbc.themes.BOOTSTRAP],add_bootstrap_links=True)
    file_upload = html.Div([
            html.Div([
            html.Div(children=[
                dcc.Upload(id='upload-data1',children=html.Div([html.A('Select Drug Interaction FOV file')]),)],
                style={'width': '16.6%', 'display': 'inline-block','borderWidth': '1px','borderStyle': 'solid','borderRadius': '5px','textAlign': 'center',},),
            html.Div(children=[
                dcc.Upload(id='upload-data2',children=html.Div([html.A('Select Whole Image population file')]),)],
                style={'width': '16.6%', 'display': 'inline-block','borderWidth': '1px','borderStyle': 'solid','borderRadius': '5px','textAlign': 'center',},),
            html.Div(children=[
                dcc.Upload(id='upload-data3',children=html.Div([html.A('Select PlateResult file')]),)],
                style={'width': '16.8%', 'display': 'inline-block','borderWidth': '1px','borderStyle': 'solid','borderRadius': '5px','textAlign': 'center',},),
            html.Div(children=[
                dcc.Upload(id='upload-data4',children=html.Div([html.A('Select AOPI 1 file')]),)],
                style={'width': '15%', 'display': 'inline-block','borderWidth': '1px','borderStyle': 'solid','borderRadius': '5px','textAlign': 'center',},),
            html.Div(children=[
                dcc.Upload(id='upload-data5',children=html.Div([html.A('Select AOPI 2 file')]),)],
                style={'width': '15%', 'display': 'inline-block','borderWidth': '1px','borderStyle': 'solid','borderRadius': '5px','textAlign': 'center',},),
                ]),
                ], style={'display': 'block'}, id='check-container')
    plot_upload = html.Div([
            html.Div([
            html.Div(children=[
                dcc.Graph(id='output-image-upload', clear_on_unhover=True,config={"displaylogo": False, 'modeBarButtonsToAdd':['zoom2d','drawopenpath','drawrect', 'eraseshape','resetViews','resetGeo'], }),
                ]),
            html.Br(),
            html.Div(children=[
            dcc.Graph(id="indicator-graphic", clear_on_unhover=True,config={"displaylogo": False,'toImageButtonOptions': {
                                                                                       'format': 'svg', 'filename': 'custom_image',
                                                                                       'height': 1000,'width': 1429,'scale': 1 }}),
            ]),
            html.Div(children=[
                html.Div(id='output-data-upload'),]),
                ])
                ], style={'display': 'block'}, id='check-container2')
    aopi = html.Div([
            html.Div([
             html.Div(children=[html.Div(id='aopi-data-block'),]),]) ], style={'display': 'block'}, id='check-container-aopi')
    app.layout = html.Div([
        dbc.Row([dbc.Col(file_upload)]),
        dcc.Upload(
            id='upload-image',
            children=html.Div([
                html.A('Select Image (jpg,png)')
            ]),
            style={
            'width': '80%',
            'height': '30px',
            'lineHeight': '30px',
            'borderWidth': '1px',
            'borderStyle': 'solid',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '2px'
            },
            multiple=True
        ),
        dcc.Download( id="download-pdf"),
        dbc.Button('PLATE REPORT', id='export-button',size="lg",color="primary",n_clicks=0),
        dbc.Row([dbc.Col(plot_upload)]),
        dbc.Row([dbc.Col(aopi)]),
        ])
        
    @app.callback(
        Output("download-pdf", 'data'),
        Input("export-button", 'n_clicks'),
        State('datatable-interactivity-3', "derived_virtual_data"),
        State('datatable-interactivity-aopi1', "derived_virtual_data"),
        State('datatable-interactivity-aopi2', "derived_virtual_data"),
        prevent_initial_call=True
        )
    def export_to_pdf(n_clicks,data,aopi1,aopi2):
        df = pd.DataFrame(data)
        aopi1 = pd.DataFrame(aopi1)
        aopi2 = pd.DataFrame(aopi2)
        aopi = aopi1['Viability %'] + aopi2['Viability %']
        aopi = round(aopi.mean() / 2,2)
        filtered_df_drug = df[df['Compound'].str.contains('Control', case=False, regex=False)]
        control = filtered_df_drug['Fail'].astype(int).sum()
        if control <5 and aopi > 25:
            res = 'Passed'
        elif control >5 and aopi > 25:
            res = 'Failed_control'
        elif control <5 and aopi < 25:
            res = 'Failed_aopi'
        else:
            res = 'Failed'
        table2_data = df.iloc[:12]
        table3_data = df.iloc[12:]
        geometry_options = {'tmargin':'0.5cm','lmargin':'0.5cm','rmargin':'0.5cm','paperwidth':'612pt','paperheight':'792pt'}
        doc = Document(geometry_options=geometry_options)
        doc.append(NoEscape(r'\centering'))
        # 
        table1 = Tabular('||p{7cm} p{2cm}||p{7cm} p{2cm}||',booktabs=True)
        #table1 = Tabular('|X X|X X|')
        table1.add_hline()
        #long_string = 'St. Jude Children’s Research Hospital Clinical Pharmacotyping Laboratory Memphis, TN  38105'              
        #table1.add_row((NoEscape(r'\parbox[t]{6.55cm}{' + long_string + '}'),'Form Number:  CPT.X.X'))
        table1.add_row(('St. Jude Children’s Research Hospital','','','Page 1 of 1'))
        table1.add_row(('Clinical Pharmacotyping Laboratory','','Form Number:  CPT.X.X',''))
        table1.add_row(('Memphis, TN  38105','','',''))
        table1.add_hline()
        table1.add_hline()
        table1.add_row((MultiColumn(4,align='||c||', data = ''),))
        table1.add_row((MultiColumn(4,align='||c||', data = LargeText(bold('Quality Control – Plate Processing Analysis'))),))
        table1.add_row((MultiColumn(4,align='||c||', data = ''),))
        table1.add_hline() 
        table_user = Tabular(' p{0.33\linewidth} p{0.33\linewidth} p{0.33\linewidth}|')
        table_user.add_row(('MRN/Plate#: ','Accession: ','Label here '))
        table_user.add_row(('','',''))
        table_user.add_row(('Name: ','QC Date: ',''))
        table_user.add_row(('','',''))
        #with doc.create(Section('',numbering=False)):
        doc.append(table1)
        doc.append(LineBreak())
        #doc.append('No space here.' + r'\!')
        #with doc.create(MiniPage(align='c')):
            #doc.append(table1)
        #with doc.create(Section('',numbering=False)):
        doc.append(table_user)
        doc.append(LineBreak())
        table2 = Tabular('|c|c|c|c|')
        table2.add_hline()
        table2.add_row(table2_data.columns)
        table2.add_hline()
        for _, row in table2_data.iterrows():
            table2.add_row(row)
        table2.add_hline()
        table3 = Tabular('|c|c|c|c|')
        table3.add_hline()
        table3.add_row(table3_data.columns)
        table3.add_hline()
        for _, row in table3_data.iterrows():
            table3.add_row(row)
        if table3_data.shape[0] == 11:
            table3.add_row(('','','',''))
        table3.add_hline()
        table_comment = Tabular('|c c c c c c c c c c c c|')
        table_comment.add_hline()
        table_comment.add_row(('Comments:','','','','','','','','','','',''))
        table_comment.add_hline()
        for i in range(0,12):
            table_comment.add_row('','','','','','','','','','','','')
        table_comment.add_hline()
        #with doc.create(Section('',numbering=False)):
        with doc.create(Tabular('c c c',booktabs=False)) as tables:
            tables.add_row(table2, table3, table_comment)
        with doc.create(Section('',numbering=False)):
            with doc.create(Figure(position='h!')) as plot:
                with doc.create(SubFigure(
                        position='b',
                        width=NoEscape(r'0.49\linewidth'))) as left_plot:
               
                    left_plot.add_image('/opt/pub/temp/mysite/report_1.png',width=NoEscape(r'\linewidth'))
                    left_plot.add_caption('Plate Status')
                with doc.create(SubFigure(
                        position='b',
                        width=NoEscape(r'0.53\linewidth'))) as right_plot:
                    right_plot.add_image('/opt/pub/temp/mysite/report_2.png',width=NoEscape(r'\linewidth'))
                    right_plot.add_caption('Plate Image')
        with doc.create(Figure(position='h!')) as index_picture:
            with doc.create(SubFigure(position='h!')) as left:
                left.add_image('/opt/pub/temp/mysite/Index2.PNG', width='240px')
        with doc.create(Figure(position='h!')) as qc_picture:
            if res == 'Passed':
                qc_picture.add_image('/opt/pub/temp/mysite/Passed.png',width='160px')
            elif res == 'Failed_control':
                qc_picture.add_image('/opt/pub/temp/mysite/Failed-controlwells.png',width='220px')
            elif res == 'Failed_aopi':
                qc_picture.add_image('/opt/pub/temp/mysite/Failed-aopi.png',width='220px')
            else:
                qc_picture.add_image('/opt/pub/temp/mysite/Failed.png',width='220px')
        pdf_filename = 'output'
        doc.generate_pdf(pdf_filename, clean_tex=True)
        if n_clicks > 0:
            return dcc.send_file('/opt/pub/temp/mysite/output.pdf')
        else:
            return n_clicks
    def parse_contents_plate(contents,columns):
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        return pd.read_csv(io.StringIO(decoded.decode('utf-8')),sep='\t',skiprows=8,usecols = columns)
    def parse_contents_columns(contents,columns):
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        return pd.read_csv(io.StringIO(decoded.decode('utf-8')),sep='\t',skiprows=9,usecols = columns)
    def parse_contents_AOPI(contents):
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        return pd.read_csv(io.StringIO(decoded.decode('utf-8')),sep='\t',skiprows=8)
    @app.callback(Output('output-data-upload', 'children'),
              [Input('upload-data1', 'contents'),
               Input('upload-data2', 'contents'),
               Input('upload-data3', 'contents'),
               
               ]
              )
    def update_output(contents1,contents2,contents3):
        if contents1 is None or contents2 is None or contents3 is None:
            return 'Upload all files'
        columns_df1 = [0,1,4,11,12,13,14,16,17,18,19,20,21,22]
        columns_df2 = [0,1,4,17,18]
        columns_df3 = [0,1,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,23,24,25,26]
        #columns_df4 = [0,1,4,15,16]
        df1 = parse_contents_columns(contents2,columns_df1)
        df2 = parse_contents_columns(contents1,columns_df2)
        df3 = parse_contents_plate(contents3,columns_df3)
        #df4 = parse_contents_columns(contents4,columns_df4)
        columns_to_convert = ['Row', 'Column','Field']
        columns_to_convert_plate = ['Row', 'Column']
        df1[columns_to_convert] = df1[columns_to_convert].astype(str)
        df2[columns_to_convert] = df2[columns_to_convert].astype(str)
        #df4[columns_to_convert] = df4[columns_to_convert].astype(str)
        df3[columns_to_convert_plate] = df3[columns_to_convert_plate].astype(str)
        df1['ID'] = df1['Row'] + df1['Column'] + df1['Field']
        df1['Well ID'] = df1['Row'] + df1['Column']
        df2['ID'] = df2['Row'] + df2['Column'] + df2['Field']
        df3['Well ID'] = df3['Row'] + df3['Column']
        #df4['ID'] = df4['Row'] + df4['Column'] + df4['Field']
        #columns_to_delete_1 =[2,3,5,6,7,8,9,10,14,22]
        columns_to_delete_2 = [0,1,2]
        #columns_to_delete_4 = [0,1,2]
        #columns_to_delete_3 = [2,3,20,21,22,26,27]
        #df1 = df1.drop(df1.columns[columns_to_delete_1], axis=1)
        df2 = df2.drop(df2.columns[columns_to_delete_2], axis=1)
        #df3 = df3.drop(df3.columns[columns_to_delete_3], axis=1)
        #df4 = df4.drop(df4.columns[columns_to_delete_4], axis=1)
        #df = pd.merge(df1, df4, how='left', on='ID')
        df = pd.merge(df1, df2, how='left', on='ID') 
        df['Status'] = ''
        for i in range(len(df)):
            if np.isnan(df.at[i,'Whole Image Population - Empty Ratio-2']):
                df.at[i,'Status'] = 'Normal'
            elif df.at[i,'Whole Image Population - Peeling Factor-2'] < 0.0001:
                df.at[i,'Status'] = 'Drug Interaction'
            else:
                df.at[i,'Status'] = 'Peeling'
        summary_table = df.pivot_table(index='Well ID', columns='Status', aggfunc='size', fill_value=0)
        if 'Peeling' in summary_table:
            summary_table = summary_table
        else:
            summary_table['Peeling'] = 0
        summary_table['Normal'] = 9-summary_table['Drug Interaction'] - summary_table['Peeling']
        summary_table = summary_table.reset_index()
        df_plate = pd.merge(df3, summary_table, how='left', on='Well ID')
        df_plate['Well Result'] = ''
        df_plate['Status'] = ''
        df_plate['Color'] = ''
        df_plate['Red Flag'] = ''
        for i in range(len(df_plate)):
            if df_plate.at[i,'Peeling'] > 3:
                df_plate.at[i,'Status'] = 'Fail'
            else:
                df_plate.at[i,'Status'] = 'Pass'
        for i in range(len(df_plate)):
            if df_plate.at[i,'Normal'] == 9:
                df_plate.at[i,'Well Result'] = 'Normal'
                df_plate.at[i,'Color'] = '1'
            elif df_plate.at[i,'Peeling Factor']<0.0001:
                if df_plate.at[i,'Whole Image Population - Peeling Factor-2 - Mean per Well'] > 0.0001 and df_plate.at[i,'Empty Ratio'] < 1:
                    df_plate.at[i,'Well Result'] = 'Peeling'
                    df_plate.at[i,'Color'] = '3'
                else:    
                    df_plate.at[i,'Well Result'] = 'Drug Interaction'
                    df_plate.at[i,'Color'] = '2'
            else:
                if df_plate.at[i,'Empty Area Population - Drug Interaction FOV - CV % per Well'] < 9500:
                    if df_plate.at[i,'Spots - Number of Objects'] < 10000 and  df_plate.at[i,'Hole Factor']<1:    
                        if df_plate.at[i,'Normal FOV - Number of Objects'] >20000:
                            df_plate.at[i,'Well Result'] = 'Drug Interaction'
                            df_plate.at[i,'Color'] = '2'
                        else:
                            df_plate.at[i,'Well Result'] = 'Peeling'
                            df_plate.at[i,'Color'] = '3'
                    else:
                        df_plate.at[i,'Well Result'] = 'Peeling'
                        df_plate.at[i,'Color'] = '3'                  
                else:
                    if df_plate.at[i,'Whole Image Population - Peeling Factor-2 - Mean per Well'] > 0.0001 and df_plate.at[i,'Empty Ratio'] < 1:
                        df_plate.at[i,'Well Result'] = 'Peeling'
                        df_plate.at[i,'Color'] = '3'
                    else:
                        df_plate.at[i,'Well Result'] = 'Drug Interaction'
                        df_plate.at[i,'Color'] = '2'
        def change_value(row):
            if 'Control' in row['Compound'] and row['Well Result'] == 'Drug Interaction':
                row['Well Result'] = 'Peeling'
                row['Color'] = 3
            return row
        df_plate = df_plate.apply(change_value, axis=1)  
        for i in range(len(df_plate)):
            if df_plate.at[i,'Well Result'] == 'Drug Interaction':
                df_plate.at[i,'Status'] = 'Pass'
            if df_plate.at[i,'Well Result'] == 'Drug Interaction': 
                if df_plate.at[i,'Peeling'] >3:
                    df_plate.at[i,'Red Flag'] = 'Yes'
                else:
                    df_plate.at[i,'Red Flag'] = "No"
            else:
                df_plate.at[i,'Red Flag'] = 'No'
        
        df_plate['FOV']= 'Normal: '+ df_plate['Normal'].astype(str) + '\n' + 'DI: ' + df_plate['Drug Interaction'].astype(str) + '\n' + 'Peeling: ' + df_plate['Peeling'].astype(str)
        df_plate['FOV']=df_plate['FOV'] + '<br>' + 'Well Result: ' + df_plate['Well Result'].astype(str)
        df_plate[columns_to_convert_plate] = df_plate[columns_to_convert_plate].astype(int)
        drug_order =  list(df_plate.loc[df_plate['Row'].isin([2,7,10]),]['Compound'].unique())
        df_plate['Compound'] = pd.Categorical(df_plate['Compound'], categories=drug_order, ordered=True) 
        df_drug = df_plate.pivot_table(index='Compound', columns='Status', aggfunc='size', fill_value=0).reset_index() 
        if 'Fail' not in df_drug:
            df_drug['Fail'] = 0
        df_drug['Status'] = df_drug.apply(lambda row: 'Failed' if row['Fail'] > 4 else 'Passed', axis=1)
        return html.Div([
            dash_table.DataTable(
                    df.to_dict('records'),
                    [{'name': i, 'id': i,"hideable":True} for i in df.columns],
                    id='datatable-interactivity-1',
                    fixed_columns={ 'data': 2},
                    page_size= 25,
                    filter_action="native",
                    filter_options = {'case':'insensitive'},
                    sort_action="native",
                    sort_mode="multi",
                    page_action="native",
                    style_header_conditional=[
                         {'if': {'column_id': 'Whole Image Population - Peeling Factor-2'},'backgroundColor': 'lightblue'},
                         {'if': {'column_id': 'Whole Image Population - Empty Ratio-2'},'backgroundColor': 'lightblue'},
                    ],
                    style_data_conditional=[
                         {'if': {'row_index': 'odd'},'backgroundColor': 'rgb(220,220,220)'},
                         {'if': {'filter_query': '{Whole Image Population - Peeling Factor-2} > 0.0001','column_id': 'Whole Image Population - Peeling Factor-2'},'backgroundColor': 'red','color':'white'},
                         {'if': {'filter_query': '{Whole Image Population - Empty Ratio-2} > 1','column_id': 'Whole Image Population - Empty Ratio-2'},'backgroundColor': 'red','color':'white'},
                         {'if': {'filter_query': '{Status} = Peeling','column_id': 'Status'},'backgroundColor': 'red','color':'white'}, 
                    ],
                    #hidden_columns=['Display',"Use for Z'",'Plane','Timepoint','Normal FOV - Number of Objects','Height [µm]','Time [s]','Cell Type','color'],
                    export_format='xlsx',export_headers='display',
                    style_table={'overflowX': 'auto','overflowY': 'auto','width':'auto' },
                    style_cell={'height': 'auto','minWidth': '50px', 'width': '180px', 'maxWidth': '180px',
        'whiteSpace': 'normal','overflow': 'hidden','font-family':'Helvetica'},
                    style_header={'backgroundColor': 'white','color': 'black','fontWeight': 'bold'},
                ),
            dash_table.DataTable(
                    df_plate.to_dict('records'),
                    columns = [{'name': i, 'id': i,"hideable":True} for i in df_plate.columns],
                    id='datatable-interactivity-2',
                    page_size= 25,
                    editable=True,
                    filter_action="native",
                    filter_options = {'case':'insensitive'},
                    sort_action="native",
                    sort_mode="multi",
                    page_action="native",
                    hidden_columns=['Color','FOV'],
                    #hidden_columns=['Display',"Use for Z'",'Plane','Timepoint','Normal FOV - Number of Objects','Height [µm]','Time [s]','Cell Type','color'],
                    export_format='xlsx',export_headers='display',
                    style_table={'overflowX': 'auto'},
                    style_cell={'height': 'auto','minWidth': '50px', 'width': '180px', 'maxWidth': '180px',
        'whiteSpace': 'normal','overflow': 'hidden','font-family':'Helvetica'},
                    style_header_conditional=[
                         {'if': {'column_id': 'Peeling Factor'},'backgroundColor': 'lightblue'},
                         {'if': {'column_id': 'Whole Image Population - Peeling Factor-2 - Mean per Well' },'backgroundColor': 'lightblue'},
                         {'if': {'column_id': 'Empty Ratio'  },'backgroundColor': 'lightblue'},
                         {'if': {'column_id': 'Empty Area Population - Drug Interaction FOV - CV % per Well'},'backgroundColor': 'lightblue'},
                         {'if': {'column_id': 'Spots - Number of Objects'},'backgroundColor': 'lightblue'},
                         {'if': {'column_id': 'Hole Factor'},'backgroundColor': 'lightblue'},
                         {'if': {'column_id': 'Normal FOV - Number of Objects'},'backgroundColor': 'lightblue'},                    
                    ],
                    style_data_conditional=[
                         {'if': {'row_index': 'odd'},'backgroundColor': 'rgb(220,220,220)'},
                         {'if': {'filter_query': '{Peeling Factor} > 0.0001','column_id': 'Peeling Factor'},'backgroundColor': 'red','color':'white'},
                         {'if': {'filter_query': '{Well Result} = Peeling','column_id': 'Well Result'},'backgroundColor': 'red','color':'white'},
                         {'if': {'filter_query': '{Whole Image Population - Peeling Factor-2 - Mean per Well}>0.0001','column_id': 'Whole Image Population - Peeling Factor-2 - Mean per Well' },'backgroundColor': 'red','color':'white'},
                         {'if': {'filter_query': '{Normal FOV - Number of Objects }<20000','column_id':'Normal FOV - Number of Objects'}, 'backgroundColor': 'red','color':'white'},
                         {'if': {'filter_query': '{Spots - Number of Objects}>1000','column_id':'Spots - Number of Objects'}, 'backgroundColor': 'red','color':'white'},
                         {'if': {'filter_query': '{Hole Factor }>1','column_id':'Hole Factor' }, 'backgroundColor': 'red','color':'white'},
                         {'if': {'filter_query': '{Empty Area Population - Drug Interaction FOV - CV % per Well}>9500','column_id':'Empty Area Population - Drug Interaction FOV - CV % per Well'}, 'backgroundColor': 'red','color':'white'},
                    ],
                    style_header={'backgroundColor': 'white','color': 'black','fontWeight': 'bold'},
                ),
            dash_table.DataTable(
                    df_drug.to_dict('records'),
                    [{'name': i, 'id': i,"hideable":True} for i in df_drug.columns], 
                    id='datatable-interactivity-3',
                    editable=True,
                    page_size= 25,
                    filter_action="native",
                    filter_options = {'case':'insensitive'},
                    sort_action="native",
                    sort_mode="multi",
                    page_action="native",
                    hidden_columns=['Color','FOV'],
                    #hidden_columns=['Display',"Use for Z'",'Plane','Timepoint','Normal FOV - Number of Objects','Height [µm]','Time [s]','Cell Type','color'],
                    export_format='xlsx',export_headers='display',
                    style_table={'overflowX': 'auto'},
                    style_cell={'height': 'auto','minWidth': '50px', 'width': '180px', 'maxWidth': '180px',
        'whiteSpace': 'normal','overflow': 'hidden','font-family':'Helvetica'},
                    style_header={'backgroundColor': 'white','color': 'black','fontWeight': 'bold'},
                    style_data_conditional=[
                        {'if': {'row_index': 'odd'},'backgroundColor': 'rgb(220,220,220)'},
                    ],
                ),
                ])
    def image_contents(contents):
        content_type, content_string = contents.split(',')
        decoded_image = base64.b64decode(content_string)
        nparr = np.frombuffer(decoded_image, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return image
        #return   html.Div([
         #   html.H5(filename),
          #  html.Img(src=contents),
           # html.Hr(),
        #])
    @app.callback(Output('output-image-upload', 'figure'),
              Input('upload-image', 'contents'),
              State('upload-image', 'filename'),
              )
    def update_image(list_of_contents,list_of_filename):
        if list_of_contents is not None:
            contents = list_of_contents[-1]
            filenames = list_of_filename[-1]
            t= 'Plate '+ filenames.split('.')[0]
            image = image_contents(contents)
            fig = go.Figure()
            if image is not None:
                fig = px.imshow(image,height = 738,width=1291.5,title = t)
                fig.update_xaxes(showticklabels=False)
                fig.update_yaxes(showticklabels=False)
                
                for i in range(1,22):
                    fig.add_annotation(x=(((image.shape[1] /21) * i ) - (image.shape[1] /42)), y=(image.shape[0] + (image.shape[0] /24)),text=str(i+2),showarrow=False,font=dict( size=20,))
                
                cordinator = ['A','B','C','D','E','F','G','J','K','L','M','N','O']
                for i in range(1,13):
                    fig.add_annotation(x=(0 - (image.shape[1] /42)), y=(((image.shape[0] /12) * i ) - (image.shape[0] /24)),text=cordinator[i],showarrow=False,font=dict( size=20,)) 
                fig.add_shape(type="rect",x0=0,y0=0,x1=((image.shape[1] /21) * 21) ,y1=image.shape[0],line=dict(color="Grey",width=3,))
                '''
                for i in range(1,21,2):
                    fig.add_shape(type="rect",x0=((image.shape[1] /21) * i),y0=0,x1=((image.shape[1] /21) * (i+2)) ,y1=image.shape[0] ,line=dict(color="Grey",width=3,))
                '''
                fig.add_shape(type="rect",x0=0,y0=0,x1=((image.shape[1] /21) * 21) ,y1=image.shape[0] /2,line=dict(color="Grey",width=3,))
                fig.update_traces(hoverinfo='none',hovertemplate=None)
                fig.update_layout(plot_bgcolor='rgba(0, 0, 0, 0)',)
                fig.update_layout(margin=dict(l=0, r=0, t=55, b=0),)
                fig.update_layout( title_x=0.5)
                fig.update_layout( title=dict(font =dict(size= 35)))
                file_path = os.path.join(os.getcwd(), 'report_2.png')
                pio.write_image(fig, file_path, format='png')
                return fig
        return {'data': [], 'layout': go.Layout()}
    
    @app.callback(Output('aopi-data-block', 'children'),
              [Input('upload-data4', 'contents'),
               Input('upload-data5', 'contents'),
               ],
               prevent_initial_call=True)
    def aopi(aopi1,aopi2):
        aopi1 = parse_contents_AOPI(aopi1)
        aopi2 = parse_contents_AOPI(aopi2)
        return html.Div([
            dash_table.DataTable(
                    aopi1.to_dict('records'),
                    [{'name': i, 'id': i,"hideable":True} for i in aopi1.columns],
                    id='datatable-interactivity-aopi1',
                    export_format='xlsx',export_headers='display',
                    style_table={'overflowX': 'auto','overflowY': 'auto','width':'auto' },
                    style_cell={'height': 'auto','minWidth': '50px', 'width': '180px', 'maxWidth': '180px',
        'whiteSpace': 'normal','overflow': 'hidden','font-family':'Helvetica'},
                    style_header={'backgroundColor': 'white','color': 'black','fontWeight': 'bold'},
                    ),
            dash_table.DataTable(
                    aopi2.to_dict('records'),
                    [{'name': i, 'id': i,"hideable":True} for i in aopi2.columns],
                    id='datatable-interactivity-aopi2',
                    export_format='xlsx',export_headers='display',
                    style_table={'overflowX': 'auto','overflowY': 'auto','width':'auto' },
                    style_cell={'height': 'auto','minWidth': '50px', 'width': '180px', 'maxWidth': '180px',
        'whiteSpace': 'normal','overflow': 'hidden','font-family':'Helvetica'},
                    style_header={'backgroundColor': 'white','color': 'black','fontWeight': 'bold'},
                    ),
           ])
    
    @app.callback(
    Output("indicator-graphic", "figure"),
    [Input('datatable-interactivity-1', "derived_virtual_data"),
    Input('datatable-interactivity-2', "derived_virtual_data"),
    Input('datatable-interactivity-3', "derived_virtual_data"),
    Input('datatable-interactivity-aopi1', "derived_virtual_data"),
    Input('datatable-interactivity-aopi2', "derived_virtual_data")
    ])
    def update_graphs(data1,data2,data3,aopi1,aopi2):
        df = pd.DataFrame(data1)
        df_plate = pd.DataFrame(data2)
        df_drug = pd.DataFrame(data3)
        aopi1 = pd.DataFrame(aopi1)
        aopi2 = pd.DataFrame(aopi2)
        cordinator = ['A','B','C','D','E','F','G','','','J','K','L','M','N','O'] 
        filtered_df_drug = df_drug[df_drug['Compound'].str.contains('Control', case=False, regex=False)]
        control = filtered_df_drug['Fail'].astype(int).sum()
        aopi = aopi1['Viability %'] + aopi2['Viability %']
        aopi = round(aopi.mean() / 2,2)
        t = 'AOPI Viability %:' + str(aopi)+ ' '  + 'Failed Control Well:' + str(control)+ ' '  +'THIS PLATE HAS PASSED QC RULES' if control <5 and aopi > 25  else 'AOPI Viability %:' + str(aopi) + " " +'Failed Control Well:' + str(control)+ ' '  + 'THIS PLATE HAS NOT PASSED QC RULES'
        fig = px.imshow(df_plate.pivot('Row', 'Column', 'Color'),zmax = 3,zmin = 1,color_continuous_scale="Blues",title= t)
        #fig = px.imshow(df_plate.pivot('Row', 'Column', 'Color'),zmax = 3,zmin = 1,color_continuous_scale="Blues")
        fig.update_layout( title_x=0.5)
        fig.update_layout( title=dict(font =dict(size= 25)))
        fig.update(data=[{'customdata': df_plate.pivot('Row', 'Column', 'FOV'),'hovertemplate': 'Coloum: %{x}<br>Row: %{y}<br>FOV: %{customdata}<br><extra></extra>'}])
        fig.update_layout(coloraxis_showscale=False)
        fig.update_layout(height=713,width=1065.5,)
        fig.update_layout(plot_bgcolor='rgba(0, 0, 0, 0)',)
        fig.update_layout(margin=dict(l=0, r=0, t=40, b=0),)
        fig.update_layout(yaxis_title=None,xaxis_title=None,font=dict(size=20),xaxis=dict(tickmode='linear'),yaxis=dict(tickmode='linear',side="left"))
        fig.update_yaxes(ticktext=cordinator,tickvals=[2,3,4,5,6,7,8,9,10,11,12,13,14,15])
        fig.update_yaxes(title_text="<b>secondary</b> yaxis title", secondary_y=True)
        fig.update_layout(coloraxis_colorbar=dict(title=None, ticktext=cordinator, tickmode='array', ticks='outside'))
        if '2' in df['Column'].values:
            fig.add_shape(type="rect",x0=1.5,y0=1.5,x1=23.5,y1=15.5)
        else:
            fig.add_shape(type="rect",x0=2.5,y0=1.5,x1=23.5,y1=15.5)
        for i in range(3,23,2):
            fig.add_vline(x=i+0.5,line_width=1)
            fig.add_vline(x=i+1.5,line_width=1, line_dash="dash")
        for i in range(2,15):
            fig.add_hline(y=i+0.5,line_width=1, line_dash="dash") 
        
        fig.add_shape(type="rect",x0=2.5, y0=5.5, x1=3.5, y1=6.5,fillcolor="Grey",),fig.add_shape(type="rect",x0=2.5, y0=13.5, x1=3.5, y1=14.5,fillcolor="Grey",)
        if '2' in df['Column'].astype(str).values:
            fig.add_vline(x=2.5,line_width=1)
            fig.add_shape(type="rect",x0=1.5, y0=7.5, x1=23.5, y1=9.5,fillcolor="White",)
        else:
            fig.add_shape(type="rect",x0=2.5, y0=7.5, x1=23.5, y1=9.5,fillcolor="White",)
        fig.add_annotation(x=3, y=8,text='Control',showarrow=False,font=dict(size=15),textangle=-25),
        fig.add_annotation(x=3, y=9,text='Control',showarrow=False,font=dict(size=15),textangle=-25)
        for i in range(4,24,2):
            d= df_plate.loc[(df_plate['Row'] == 2) & (df_plate['Column'] == i),'Compound' ].values[0]
            fig.add_annotation(x=i+0.5, y=8,text=d,showarrow=False,font=dict(size=15),textangle=-25)
            if df_drug.loc[df_drug['Compound'] == d,'Status'].values[0] == 'Failed':
                fig.add_shape(type="rect",x0=i-0.5,y0=1.5,x1=i+1.5,y1=7.5, line=dict(color="Red",width = 4),)
        for i in range(2,16):
            fig.add_annotation(x=23+1, y=i,text=cordinator[i-1],showarrow=False,font=dict(size=20))
        for i in range(4,24,2):
            d= df_plate.loc[(df_plate['Row'] == 10) & (df_plate['Column'] == i),'Compound' ].values[0]
            fig.add_annotation(x=i+0.5, y=9,text=d,showarrow=False,font=dict(size=15),textangle=-25)
            if df_drug.loc[df_drug['Compound'] == d,'Status'].values[0] == 'Failed':
                fig.add_shape(type="rect",x0=i-0.5,y0=9.5,x1=i+1.5,y1=15.5,line=dict(color="Red",width=4))
        for i in range(len(df_plate)):
            if df_plate.at[i,'Red Flag'] == 'Yes':
                fig.add_shape(type="rect",x0=df_plate.at[i,'Column']-0.5,y0=df_plate.at[i,'Row']-0.5,x1=df_plate.at[i,'Column']+0.5,y1=df_plate.at[i,'Row']+0.5,line=dict(color="Red",width=4,dash='dot'),) 
        file_path = os.path.join(os.getcwd(), 'report_1.png')
        pio.write_image(fig, file_path, format='png')
        return fig
    context = {}
    return render(request, 'catalog/image.html',context)

@login_required(login_url='/accounts/login/')    
def qc2(request): 
    app = DjangoDash('app_qc2',external_stylesheets=[dbc.themes.BOOTSTRAP],add_bootstrap_links=True)
    file_upload = html.Div([
            html.Div([
            #html.Div(children=[
                #dcc.Upload(id='upload-data1',children=html.Div([html.A('Select Objects_Population - Cells')]),)],
                #style={'width': '16.6%', 'display': 'inline-block','borderWidth': '1px','borderStyle': 'solid','borderRadius': '5px','textAlign': 'center',},),
            #html.Div(children=[
                #dcc.Upload(id='upload-data2',children=html.Div([html.A('Select Objects_Population - CyQuant')]),)],
                #style={'width': '16.6%', 'display': 'inline-block','borderWidth': '1px','borderStyle': 'solid','borderRadius': '5px','textAlign': 'center',},),
            html.Div(children=[
                dcc.Upload(id='upload-data3',children=html.Div([html.A('Select PlateResult file')]),)],
                style={'width': '80%', 'display': 'inline-block','borderWidth': '1px','borderStyle': 'solid','borderRadius': '5px','textAlign': 'center',},),
                ]),
                ], style={'display': 'block'}, id='check-container')
    plot_upload = html.Div([
            html.Div([
            html.Div(children=[
                dcc.Graph(id='output-image-upload', clear_on_unhover=True,config={"displaylogo": False, 'modeBarButtonsToAdd':['zoom2d','drawopenpath','drawrect', 'eraseshape','resetViews','resetGeo'], }),
                ]),
            html.Br(),
            html.Div(children=[
            dcc.Graph(id="indicator-graphic", clear_on_unhover=True,config={"displaylogo": False,'toImageButtonOptions': {
                                                                                       'format': 'svg', 'filename': 'custom_image',
                                                                                       'height': 1000,'width': 1429,'scale': 1 }}),
            ]),
            html.Div(children=[
                html.Div(id='output-data-upload'),]),
                ])
                ], style={'display': 'block'}, id='check-container2')
    app.layout = html.Div([
        dbc.Row([dbc.Col(file_upload)]),
        dcc.Upload(
            id='upload-image',
            children=html.Div([
                html.A('Select Image (jpg,png)')
            ]),
            style={
            'width': '80%',
            'height': '30px',
            'lineHeight': '30px',
            'borderWidth': '1px',
            'borderStyle': 'solid',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '2px'
            },
            multiple=True
        ),
        dbc.Row([dbc.Col(plot_upload)]),
        ])
    def parse_contents_plate(contents):
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        return pd.read_csv(io.StringIO(decoded.decode('utf-8')),sep='\t',skiprows=8)
    def parse_contents_columns(contents):
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        return pd.read_csv(io.StringIO(decoded.decode('utf-8')),sep='\t',skiprows=9)
    @app.callback(Output('output-data-upload', 'children'),
              [
               
               Input('upload-data3', 'contents'),
               ]
              )
    def update_output(contents3):
        #df2 = parse_contents_columns(contents2)
        #df1 = parse_contents_columns(contents1)
        df3 = parse_contents_plate(contents3)
        columns_to_convert = ['Row', 'Column','Field']
        columns_to_convert_plate = ['Row', 'Column']
        #df1[columns_to_convert] = df1[columns_to_convert].astype(str)
        #df2[columns_to_convert] = df2[columns_to_convert].astype(str)
        df3[columns_to_convert_plate] = df3[columns_to_convert_plate].astype(str)
        #df1['ID'] = df1['Row'] + df1['Column'] + df1['Field']
        #df1['Well ID'] = df1['Row'] + df1['Column']
        #df2['ID'] = df2['Row'] + df2['Column']
        df3['Well ID'] = df3['Row'] + df3['Column']
        #df3['Peeling Ratio'] = df3['Emptyness - Empty CyQuant AREA Area [µm²] - Sum per Well'] / df3['CELL AREA'] 
        df3['Var1'] = abs(df3['Emptyness - Empty CyQuant Region Centroid X in Well [µm] - Mean per Well'])+abs(df3['Emptyness - Empty CyQuant Region Centroid Y in Well [µm] - Mean per Well'])
        df3['Var2'] = df3['Peeled Layer - Peeled Region Area [µm²] - Max per Well'] / df3['Cell Layer - Cell Region Area [µm²] - Max per Well']
        df3['Status'] = ''
        df3['Color'] = ''
        df3 = round(df3,5)
        #df3['Red Flag'] = ''
        #df3['Regression'] = ''
        for i in range(len(df3)):
            if df3.at[i,'Whole Image (global) - Emptyness (global) overlap Overlap [%] - Mean per Well'] > 35 and (df3.at[i,'Var1'] > 180 and df3.at[i,'Var1'] < 250 ) and df3.at[i,'Var2'] > 1 :
                df3.at[i,'Status'] = 'Fail'
                df3.at[i,'Color'] = '4'
            elif df3.at[i,'Whole Image (global) - Emptyness (global) overlap Overlap [%] - Mean per Well'] < 35 and (df3.at[i,'Var1'] > 250 and df3.at[i,'Var1'] < 500 ) and df3.at[i,'Var2'] < 1:
                df3.at[i,'Status'] = 'Pass'
                df3.at[i,'Color'] = '1'
            else:
                df3.at[i,'Status'] = 'Pass'
                df3.at[i,'Color'] = '1'
                
        '''
        for i in range(len(df3)):
            if df3.at[i,'Normal - Number of Objects'] == 1:
                df3.at[i,'Status'] = 'Normal'
                df3.at[i,'Color'] = '1'
            elif df3.at[i,'Peeling - Number of Objects'] == 1:
                df3.at[i,'Status'] = 'Peeling'
                df3.at[i,'Color'] = '3'
            elif df3.at[i,'Drug Interaction - Number of Objects'] == 1:
                df3.at[i,'Status'] = 'Drug Interaction'
                df3.at[i,'Color'] = '2'
            else:
                df3.at[i,'Status'] = 'Normal'
                df3.at[i,'Color'] = '1'
        
        for i in range(len(df3)):
            if (df3.at[i,'Cells (global) - Regression A-B - Mean per Well'] > -1 and df3.at[i,'Cells (global) - Regression A-B - Mean per Well'] < 1 and df3.at[i,'Status'] == 'Peeling' ):
                df3.at[i,'Red Flag'] = 'Yes' 
                df3.at[i,'Regression'] = df3.at[i,'Status']  + '\n' + 'Regression: '+df3.at[i,'Cells (global) - Regression A-B - Mean per Well'].astype(str) + '\n' + 'Peeling Ratio: '+ df3.at[i,'Peeling Ratio'].astype(str)
            else:
                df3.at[i,'Red Flag'] = 'No' 
                df3.at[i,'Regression'] = df3.at[i,'Status']  + '\n' + 'Regression: '+df3.at[i,'Cells (global) - Regression A-B - Mean per Well'].astype(str) + '\n' + 'Peeling Ratio: '+ df3.at[i,'Peeling Ratio'].astype(str)
        '''
        return html.Div([
            dash_table.DataTable(
                    df3.to_dict('records'),
                    [{'name': i, 'id': i,"hideable":True} for i in df3.columns],
                    id='datatable-interactivity-3',
                    fixed_columns={ 'data': 2},
                    page_size= 25,
                    editable=True,
                    filter_action="native",
                    filter_options = {'case':'insensitive'},
                    sort_action="native",
                    sort_mode="multi",
                    page_action="native",
                    #hidden_columns=['Regression'],
                    export_format='xlsx',export_headers='display',
                    style_table={'overflowX': 'auto','overflowY': 'auto','width':'auto' },
                    style_cell={'height': 'auto','minWidth': '50px', 'width': '180px', 'maxWidth': '180px',
        'whiteSpace': 'normal','overflow': 'hidden','font-family':'Helvetica'},
                    style_header={'backgroundColor': 'white','color': 'black','fontWeight': 'bold'},
                ),
                ])
    def image_contents(contents):
        content_type, content_string = contents.split(',')
        decoded_image = base64.b64decode(content_string)
        nparr = np.frombuffer(decoded_image, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return image
    @app.callback(Output('output-image-upload', 'figure'),
              Input('upload-image', 'contents'),
              State('upload-image', 'filename'),
              )
    def update_image(list_of_contents,list_of_filename):
        if list_of_contents is not None:
            contents = list_of_contents[-1]
            filenames = list_of_filename[-1]
            t= 'Plate '+ filenames.split('.')[0]
            image = image_contents(contents)
            fig = go.Figure()
            if image is not None:
                fig = px.imshow(image,height = 738,width=1291.5,title = t)
                fig.update_xaxes(showticklabels=False)
                fig.update_yaxes(showticklabels=False)
                for i in range(1,22):
                    fig.add_annotation(x=(((image.shape[1] /21) * i ) - (image.shape[1] /42)), y=(image.shape[0] + (image.shape[0] /24)),text=str(i+2),showarrow=False,font=dict( size=20,))
                cordinator = ['1','2','3','4','5','6','7','10','11','12','13','14','15']
                for i in range(1,13):
                    fig.add_annotation(x=(0 - (image.shape[1] /42)), y=(((image.shape[0] /12) * i ) - (image.shape[0] /24)),text=cordinator[i],showarrow=False,font=dict( size=20,)) 
                fig.add_shape(type="rect",x0=0,y0=0,x1=((image.shape[1] /21) * 21) ,y1=image.shape[0],line=dict(color="Grey",width=3,))
                for i in range(1,21,2):
                    fig.add_shape(type="rect",x0=((image.shape[1] /21) * i),y0=0,x1=((image.shape[1] /21) * (i+2)) ,y1=image.shape[0] ,line=dict(color="Grey",width=3,))
                fig.add_shape(type="rect",x0=0,y0=0,x1=((image.shape[1] /21) * 21) ,y1=image.shape[0] /2,line=dict(color="Grey",width=3,))
                fig.update_traces(hoverinfo='none',hovertemplate=None)
                fig.update_layout(plot_bgcolor='rgba(0, 0, 0, 0)',)
                fig.update_layout(margin=dict(l=0, r=0, t=55, b=0),)
                fig.update_layout( title_x=0.5)
                fig.update_layout( title=dict(font =dict(size= 35)))
                file_path = os.path.join(os.getcwd(), 'report_2.png')
                pio.write_image(fig, file_path, format='png')
                return fig
        return {'data': [], 'layout': go.Layout()}
    @app.callback(
    Output("indicator-graphic", "figure"),
    [Input('datatable-interactivity-3', "derived_virtual_data"),]
    )    
    def update_graphs(data1):
        df = pd.DataFrame(data1)
        columns_to_convert = ['Row', 'Column']
        df[columns_to_convert] = df[columns_to_convert].astype(int)
        discrete= {4: 'rgb(139,0,0)'}
        fig = px.imshow(df.pivot('Row', 'Column', 'Color'),zmax = 4,zmin =1 ,color_continuous_scale="Blues")
        fig.update_layout(coloraxis_showscale=False)
        fig.update(data=[{'customdata': df.pivot('Row', 'Column', 'Well ID'),'hovertemplate': 'Coloum: %{x}<br>Row: %{y}<br>Well ID: %{customdata}<br><extra></extra>'}])
        fig.update_layout(height=713,width=1065.5,)
        fig.update_layout(plot_bgcolor='rgba(0, 0, 0, 0)',)
        fig.update_layout(margin=dict(l=0, r=0, t=40, b=0),)
        fig.add_shape(type="rect",x0=2.5,y0=1.5,x1=23.5,y1=15.5)
        fig.update_layout(font=dict(size=20),xaxis=dict(tickmode='linear'),yaxis=dict(tickmode='linear',side="left"))
        for i in range(3,23,2):
            fig.add_vline(x=i+0.5,line_width=1)
            fig.add_vline(x=i+1.5,line_width=1, line_dash="dash")
        for i in range(2,15):
            fig.add_hline(y=i+0.5,line_width=1, line_dash="dash") 
        fig.add_shape(type="rect",x0=2.5, y0=5.5, x1=3.5, y1=6.5,fillcolor="Grey",),fig.add_shape(type="rect",x0=2.5, y0=13.5, x1=3.5, y1=14.5,fillcolor="Grey",)
        fig.add_shape(type="rect",x0=2.5, y0=7.5, x1=23.5, y1=9.5,fillcolor="White",)
        fig.add_annotation(x=3, y=8,text='Control',showarrow=False,font=dict(size=15),textangle=-25),fig.add_annotation(x=3, y=9,text='Control',showarrow=False,font=dict(size=15),textangle=-25)
        for i in range(4,24,2):
            d= df.loc[(df['Row'] == 2) & (df['Column'] == i),'Compound' ].values[0]
            fig.add_annotation(x=i+0.5, y=8,text=d,showarrow=False,font=dict(size=15),textangle=-25)
        for i in range(4,24,2):
            d= df.loc[(df['Row'] == 10) & (df['Column'] == i),'Compound' ].values[0]
            fig.add_annotation(x=i+0.5, y=9,text=d,showarrow=False,font=dict(size=15),textangle=-25)
        #for i in range(len(df)):
        #    if df.at[i,'Red Flag'] == 'Yes':
        #        fig.add_shape(type="rect",x0=df.at[i,'Column']-0.5,y0=df.at[i,'Row']-0.5,x1=df.at[i,'Column']+0.5,y1=df.at[i,'Row']+0.5,line=dict(color="Red",width=4,dash='dot'),) 
        return fig
    context = {}
    return render(request, 'catalog/qc2.html',context)   
@login_required(login_url='/accounts/login/')    
def qc3(request): 
    app = DjangoDash('app_qc3',external_stylesheets=[dbc.themes.BOOTSTRAP],add_bootstrap_links=True)
    file_upload = html.Div([
            html.Div([
            #html.Div(children=[
                #dcc.Upload(id='upload-data1',children=html.Div([html.A('Select Objects_Population - Cells')]),)],
                #style={'width': '16.6%', 'display': 'inline-block','borderWidth': '1px','borderStyle': 'solid','borderRadius': '5px','textAlign': 'center',},),
            #html.Div(children=[
                #dcc.Upload(id='upload-data2',children=html.Div([html.A('Select Objects_Population - CyQuant')]),)],
                #style={'width': '16.6%', 'display': 'inline-block','borderWidth': '1px','borderStyle': 'solid','borderRadius': '5px','textAlign': 'center',},),
            html.Div(children=[
                dcc.Upload(id='upload-data3',children=html.Div([html.A('Select PlateResult file')]),)],
                style={'width': '80%', 'display': 'inline-block','borderWidth': '1px','borderStyle': 'solid','borderRadius': '5px','textAlign': 'center',},),
                ]),
                ], style={'display': 'block'}, id='check-container')
    plot_upload = html.Div([
            html.Div([
            html.Div(children=[
                dcc.Graph(id='output-image-upload', clear_on_unhover=True,config={"displaylogo": False, 'modeBarButtonsToAdd':['zoom2d','drawopenpath','drawrect', 'eraseshape','resetViews','resetGeo'], }),
                ]),
            html.Br(),
            html.Div(children=[
            dcc.Graph(id="indicator-graphic", clear_on_unhover=True,config={"displaylogo": False,'toImageButtonOptions': {
                                                                                       'format': 'svg', 'filename': 'custom_image',
                                                                                       'height': 1000,'width': 1429,'scale': 1 }}),
            ]),
            html.Div(children=[
                html.Div(id='output-data-upload'),]),
                ])
                ], style={'display': 'block'}, id='check-container2')
    app.layout = html.Div([
        dbc.Row([dbc.Col(file_upload)]),
        dcc.Upload(
            id='upload-image',
            children=html.Div([
                html.A('Select Image (jpg,png)')
            ]),
            style={
            'width': '80%',
            'height': '30px',
            'lineHeight': '30px',
            'borderWidth': '1px',
            'borderStyle': 'solid',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '2px'
            },
            multiple=True
        ),
        dbc.Row([dbc.Col(plot_upload)]),
        ])
    def parse_contents_plate(contents):
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        return pd.read_csv(io.StringIO(decoded.decode('utf-8')),sep='\t',skiprows=8)
    def parse_contents_columns(contents):
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        return pd.read_csv(io.StringIO(decoded.decode('utf-8')),sep='\t',skiprows=9)
    @app.callback(Output('output-data-upload', 'children'),
              [
               
               Input('upload-data3', 'contents'),
               ]
              )
    def update_output(contents3):
        #df2 = parse_contents_columns(contents2)
        #df1 = parse_contents_columns(contents1)
        df3 = parse_contents_plate(contents3)
        columns_to_convert = ['Row', 'Column','Field']
        columns_to_convert_plate = ['Row', 'Column']
        #df1[columns_to_convert] = df1[columns_to_convert].astype(str)
        #df2[columns_to_convert] = df2[columns_to_convert].astype(str)
        df3[columns_to_convert_plate] = df3[columns_to_convert_plate].astype(str)
        #df1['ID'] = df1['Row'] + df1['Column'] + df1['Field']
        #df1['Well ID'] = df1['Row'] + df1['Column']
        #df2['ID'] = df2['Row'] + df2['Column']
        df3['Well ID'] = df3['Row'] + df3['Column']
        df3 = df3.round(5)
        df3['Status'] = ''  
        df3['Color'] = ''
        df3['AbsSum of XY Empty'] = abs(df3['Emptyness - Empty CyQuant Region Centroid X in Well [µm] - Mean per Well']) + abs(df3['Emptyness - Empty CyQuant Region Centroid Y in Well [µm] - Mean per Well'])
        df3['AbsSum of XY Severe'] = (abs(df3['Severe Peeling - Empty CyQuant Region Centroid X in Well [µm] - Mean per Well']) * abs(df3['Severe Peeling - Empty CyQuant Region Centroid X in Well [µm] - CV % per Well'])) + (abs(df3['Severe Peeling - Empty CyQuant Region Centroid Y in Well [µm] - Mean per Well']) * abs(df3['Severe Peeling - Empty CyQuant Region Centroid Y in Well [µm] - CV % per Well']))
        df3['AbsSum of XY Minor'] = abs(df3['Minor Peeling - Empty CyQuant Region Centroid X in Well [µm] - Mean per Well'])*abs(df3['Minor Peeling - Empty CyQuant Region Centroid X in Well [µm] - CV % per Well']) + abs(df3['Minor Peeling - Empty CyQuant Region Centroid Y in Well [µm] - Mean per Well']) *abs(df3['Minor Peeling - Empty CyQuant Region Centroid Y in Well [µm] - CV % per Well'])
        df3['AbsSum of XY Normal'] = abs(df3['Normal - Empty CyQuant Region Centroid X in Well [µm] - Mean per Well'])*abs(df3['Normal - Empty CyQuant Region Centroid X in Well [µm] - CV % per Well']) + abs(df3['Normal - Empty CyQuant Region Centroid Y in Well [µm] - Mean per Well'])*abs(df3['Normal - Empty CyQuant Region Centroid Y in Well [µm] - CV % per Well'])
        
        df3['XY Sum Severe&Normal']= df3['AbsSum of XY Severe'] + df3['AbsSum of XY Normal']
        df3['XY sum Severe&Normal divided by AbsSum of XY Empty'] = df3['XY Sum Severe&Normal'] / df3['AbsSum of XY Empty']
        df3 = df3.round(5)
        for i in range(len(df3)):
            if df3.at[i,'Whole Image (global) - Emptyness (global) overlap Overlap [%] - Mean per Well'] > 90:
                df3.at[i,'Status'] = 'Fail'
                df3.at[i,'Color'] = '4'
            elif df3.at[i,'XY sum Severe&Normal divided by AbsSum of XY Empty'] > 300 and df3.at[i,'AbsSum of XY Severe'] > 50000:
                if df3.at[i,'AbsSum of XY Empty'] >450:
                    df3.at[i,'Status'] = 'Red Flag'
                    df3.at[i,'Color'] = '2'
                else:
                    df3.at[i,'Status'] = 'Pass'
                    df3.at[i,'Color'] = '1'
            else:
                df3.at[i,'Status'] = 'Fail'
                df3.at[i,'Color'] = '4'
        '''
        for i in range(len(df3)):
            if df3.at[i,'Normal - Number of Objects'] == 1:
                df3.at[i,'Status'] = 'Normal'
                df3.at[i,'Color'] = '1'
            elif df3.at[i,'Peeling - Number of Objects'] == 1:
                df3.at[i,'Status'] = 'Peeling'
                df3.at[i,'Color'] = '3'
            elif df3.at[i,'Drug Interaction - Number of Objects'] == 1:
                df3.at[i,'Status'] = 'Drug Interaction'
                df3.at[i,'Color'] = '2'
            else:
                df3.at[i,'Status'] = 'Red Flag'
                df3.at[i,'Color'] = '4'
        '''
        return html.Div([
            dash_table.DataTable(
                    df3.to_dict('records'),
                    [{'name': i, 'id': i,"hideable":True} for i in df3.columns],
                    id='datatable-interactivity-3',
                    fixed_columns={ 'data': 2},
                    page_size= 25,
                    filter_action="native",
                    filter_options = {'case':'insensitive'},
                    sort_action="native",
                    sort_mode="multi",
                    page_action="native",
                    hidden_columns=['Regression'],
                    export_format='xlsx',export_headers='display',
                    style_table={'overflowX': 'auto','overflowY': 'auto','width':'auto' },
                    style_cell={'height': 'auto','minWidth': '50px', 'width': '180px', 'maxWidth': '180px',
        'whiteSpace': 'normal','overflow': 'hidden','font-family':'Helvetica'},
                    style_header={'backgroundColor': 'white','color': 'black','fontWeight': 'bold'},
                ),
                ])
    def image_contents(contents):
        content_type, content_string = contents.split(',')
        decoded_image = base64.b64decode(content_string)
        nparr = np.frombuffer(decoded_image, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return image
    @app.callback(Output('output-image-upload', 'figure'),
              Input('upload-image', 'contents'),
              State('upload-image', 'filename'),
              )
    def update_image(list_of_contents,list_of_filename):
        if list_of_contents is not None:
            contents = list_of_contents[-1]
            filenames = list_of_filename[-1]
            t= 'Plate '+ filenames.split('.')[0]
            image = image_contents(contents)
            fig = go.Figure()
            if image is not None:
                fig = px.imshow(image,height = 738,width=1291.5,title = t)
                fig.update_xaxes(showticklabels=False)
                fig.update_yaxes(showticklabels=False)
                for i in range(1,22):
                    fig.add_annotation(x=(((image.shape[1] /21) * i ) - (image.shape[1] /42)), y=(image.shape[0] + (image.shape[0] /24)),text=str(i+2),showarrow=False,font=dict( size=20,))
                cordinator = ['1','2','3','4','5','6','7','10','11','12','13','14','15']
                for i in range(1,13):
                    fig.add_annotation(x=(0 - (image.shape[1] /42)), y=(((image.shape[0] /12) * i ) - (image.shape[0] /24)),text=cordinator[i],showarrow=False,font=dict( size=20,)) 
                fig.add_shape(type="rect",x0=0,y0=0,x1=((image.shape[1] /21) * 21) ,y1=image.shape[0],line=dict(color="Grey",width=3,))
                for i in range(1,21,2):
                    fig.add_shape(type="rect",x0=((image.shape[1] /21) * i),y0=0,x1=((image.shape[1] /21) * (i+2)) ,y1=image.shape[0] ,line=dict(color="Grey",width=3,))
                fig.add_shape(type="rect",x0=0,y0=0,x1=((image.shape[1] /21) * 21) ,y1=image.shape[0] /2,line=dict(color="Grey",width=3,))
                fig.update_traces(hoverinfo='none',hovertemplate=None)
                fig.update_layout(plot_bgcolor='rgba(0, 0, 0, 0)',)
                fig.update_layout(margin=dict(l=0, r=0, t=55, b=0),)
                fig.update_layout( title_x=0.5)
                fig.update_layout( title=dict(font =dict(size= 35)))
                file_path = os.path.join(os.getcwd(), 'report_2.png')
                pio.write_image(fig, file_path, format='png')
                return fig
        return {'data': [], 'layout': go.Layout()}
    @app.callback(
    Output("indicator-graphic", "figure"),
    [Input('datatable-interactivity-3', "derived_virtual_data"),]
    )    
    def update_graphs(data1):
        df = pd.DataFrame(data1)
        columns_to_convert = ['Row', 'Column']
        df[columns_to_convert] = df[columns_to_convert].astype(int)
        discrete= {4: 'rgb(139,0,0)'}
        fig = px.imshow(df.pivot('Row', 'Column', 'Color'),zmax = 4,zmin =1 ,color_continuous_scale="Blues")
        fig.update_layout(coloraxis_showscale=False)
        fig.update(data=[{'customdata': df.pivot('Row', 'Column', 'Status'),'hovertemplate': 'Coloum: %{x}<br>Row: %{y}<br>Status: %{customdata}<br><extra></extra>'}])
        fig.update_layout(height=713,width=1065.5,)
        fig.update_layout(plot_bgcolor='rgba(0, 0, 0, 0)',)
        fig.update_layout(margin=dict(l=0, r=0, t=40, b=0),)
        fig.add_shape(type="rect",x0=2.5,y0=1.5,x1=23.5,y1=15.5)
        fig.update_layout(font=dict(size=20),xaxis=dict(tickmode='linear'),yaxis=dict(tickmode='linear',side="left"))
        for i in range(3,23,2):
            fig.add_vline(x=i+0.5,line_width=1)
            fig.add_vline(x=i+1.5,line_width=1, line_dash="dash")
        for i in range(2,15):
            fig.add_hline(y=i+0.5,line_width=1, line_dash="dash") 
        fig.add_shape(type="rect",x0=2.5, y0=5.5, x1=3.5, y1=6.5,fillcolor="Grey",),fig.add_shape(type="rect",x0=2.5, y0=13.5, x1=3.5, y1=14.5,fillcolor="Grey",)
        fig.add_shape(type="rect",x0=2.5, y0=7.5, x1=23.5, y1=9.5,fillcolor="White",)
        fig.add_annotation(x=3, y=8,text='Control',showarrow=False,font=dict(size=15),textangle=-25),fig.add_annotation(x=3, y=9,text='Control',showarrow=False,font=dict(size=15),textangle=-25)
        for i in range(4,24,2):
            d= df.loc[(df['Row'] == 2) & (df['Column'] == i),'Compound' ].values[0]
            fig.add_annotation(x=i+0.5, y=8,text=d,showarrow=False,font=dict(size=15),textangle=-25)
        for i in range(4,24,2):
            d= df.loc[(df['Row'] == 10) & (df['Column'] == i),'Compound' ].values[0]
            fig.add_annotation(x=i+0.5, y=9,text=d,showarrow=False,font=dict(size=15),textangle=-25)
        ''' 
        for i in range(len(df)):
            if df.at[i,'Red Flag'] == 'Yes':
                fig.add_shape(type="rect",x0=df.at[i,'Column']-0.5,y0=df.at[i,'Row']-0.5,x1=df.at[i,'Column']+0.5,y1=df.at[i,'Row']+0.5,line=dict(color="Red",width=4,dash='dot'),) 
        '''
        return fig
    context = {}
    return render(request, 'catalog/qc3.html',context)                           
@login_required(login_url='/accounts/login/')
def liver (request):        
    adata = sc.read_h5ad('static/adata_liver.h5ad') 
    df = pd.DataFrame(adata.X.A,index=adata.obs_names,columns = adata.var.features)
    df['Treatment'] = adata.obs.treatment
    df['Sample'] = adata.obs['orig.ident']
    df['Cell Type'] = adata.obs.celltypist3
    df[['UMAP1', 'UMAP2']] = pd.DataFrame(adata.obsm['X_umap'],index=adata.obs_names)
    dropdown = list(adata.var.features)
    dropdown2 = ['ALL'] + list(df['Cell Type'].unique())
    app = DjangoDash('app_liver',external_stylesheets=[dbc.themes.BOOTSTRAP],add_bootstrap_links=True)
    col = [{"label": i , "value": i } for i in dropdown]
    '''
    table_control = html.Div([ 
            html.Div (children=[html.Button('Plot', id='editing-columns-button', n_clicks=0)],style={'width': '92%', 'display': 'inline-block'}),
            ])
    '''
    app.layout = html.Div([
        html.Div([dcc.Dropdown(dropdown2,['ALL'],multi=True,placeholder='Select Cell Type',id='cell-type',style={'width':'50%'})]),
        html.Div([dcc.Dropdown(multi=False,placeholder='Enter Gene Name',id='editing-columns-name',style={'width':'50%'})]),
        html.Div([dcc.Dropdown(multi=False,options=[1,2,3,4,5,6,7,8,9,10],value=5,id='point-size',style={'width':'50%'})]),
        html.Div([dcc.Checklist(id='nonZero',options=['Remove Zero'],value='',style={'width': '10%',})]),
        #dbc.Row([dbc.Col(table_control)]),
        html.Div([
        dcc.Graph(id="indicator-graphic1",config={"displaylogo": False,'toImageButtonOptions':{
                                                                                       'format': 'svg', 'filename': 'custom_image',
                                                                                       'height': 700,'width': 1000,'scale': 1 }}),
        dcc.Graph(id="indicator-graphic2",config={"displaylogo": False,'toImageButtonOptions': {
                                                                                       'format': 'svg', 'filename': 'custom_image',
                                                                                       'height': 700,'width': 1000,'scale': 1 }}),
                  ],style={'display': 'flex'}),
        html.Div([
        dcc.Graph(id="indicator-graphic3",config={"displaylogo": False,'toImageButtonOptions':{
                                                                                       'format': 'svg', 'filename': 'custom_image',
                                                                                       'height': 700,'width': 1000,'scale': 1 }}),
        dcc.Graph(id="indicator-graphic4",config={"displaylogo": False,'toImageButtonOptions': {
                                                                                       'format': 'svg', 'filename': 'custom_image',
                                                                                       'height': 700,'width': 1000,'scale': 1 }}),
                  ],style={'display': 'flex'}),
        dcc.Graph(id="indicator-graphic5",config={"displaylogo": False,'toImageButtonOptions': {
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
        return [o for o in col if search_value in o["label"]]
    @app.callback(
    Output('indicator-graphic1', "figure"),
    Input('cell-type', "value"),
    #Input('editing-columns-button', "n_clicks"),
    Input('point-size', "value"),
    Input('editing-columns-name', 'value'),
    Input('nonZero', 'value'),
    )
    def update_graphs1(celltype,psize,value,zero):
        fig = []
        if 'ALL' in celltype:
            if 'Remove Zero' in zero:
                fig = px.scatter(df[df[value] != 0], x="UMAP1", y="UMAP2",color = value,
                             color_continuous_scale = 'balance')
            else:
                fig = px.scatter(df, x="UMAP1", y="UMAP2",color = value,
                             color_continuous_scale = 'balance')
        else:
            if 'Remove Zero' in zero:
                fig = px.scatter(df[df[value] != 0][df[df[value] != 0]['Cell Type'].isin(celltype)], 
                             x="UMAP1", y="UMAP2",color = value,
                             color_continuous_scale = 'balance')
            else:
                fig = px.scatter(df[df['Cell Type'].isin(celltype)], x="UMAP1", y="UMAP2",
                              color = value,color_continuous_scale = 'balance')
        fig.update_traces(marker=dict(size=psize))
        fig.update_traces(hoverinfo='none',hovertemplate=None)
        fig.update_layout(plot_bgcolor='rgba(0, 0, 0, 0)',)
        fig.update_layout(height = 600,width=600 )    
        return fig    
    @app.callback(
    Output('indicator-graphic2', "figure"),
    Input('cell-type', "value"),
    #Input('editing-columns-button', "n_clicks"),
    Input('point-size', "value"),
    Input('editing-columns-name', 'value'),
    Input('nonZero', 'value'),
    )
    def update_graphs2(celltype,psize,value,zero):
        fig = []
        if 'ALL' in celltype:
            if 'Remove Zero' in zero:
                fig = px.scatter(df[df[value] != 0], x="UMAP1", y="UMAP2",color = 'Cell Type')
            else:
                fig = px.scatter(df, x="UMAP1", y="UMAP2",color = 'Cell Type')
        else:
            if 'Remove Zero' in zero:
                fig = px.scatter(df[df[value] != 0][df[df[value] != 0]['Cell Type'].isin(celltype)], 
                                 x="UMAP1", y="UMAP2",color = 'Cell Type')
            else:
                fig = px.scatter(df[df['Cell Type'].isin(celltype)], x="UMAP1", y="UMAP2",color = 'Cell Type')
        fig.update_traces(marker=dict(size=psize))
        fig.update_traces(hoverinfo='none',hovertemplate=None)
        fig.update_layout(plot_bgcolor='rgba(0, 0, 0, 0)',)
        fig.update_layout(height = 600,width=700 )
        return fig
    @app.callback(
    Output('indicator-graphic3', "figure"),
    Input('cell-type', "value"),
    #Input('editing-columns-button', "n_clicks"),
    Input('point-size', "value"),
    Input('editing-columns-name', 'value'),
    Input('nonZero', 'value'),
    )
    def update_graphs3(celltype,psize,value,zero):
        fig = []
        if 'ALL' in celltype:
            if 'Remove Zero' in zero:
                fig = px.scatter(df[df[value] != 0], x="UMAP1", y="UMAP2",color = 'Treatment')
            else:
                fig = px.scatter(df, x="UMAP1", y="UMAP2",color = 'Treatment')
        else:
            if 'Remove Zero' in zero:
                fig = px.scatter(df[df[value] != 0][df[df[value] != 0]['Cell Type'].isin(celltype)],
                                 x="UMAP1", y="UMAP2",color = 'Treatment')
            else:
                fig = px.scatter(df[df['Cell Type'].isin(celltype)], x="UMAP1", y="UMAP2",color = 'Treatment')
        fig.update_traces(marker=dict(size=psize))
        fig.update_traces(hoverinfo='none',hovertemplate=None)
        fig.update_layout(plot_bgcolor='rgba(0, 0, 0, 0)',)
        fig.update_layout(height = 600,width=699)
        return fig
    @app.callback(
    Output('indicator-graphic4', "figure"),
    Input('cell-type', "value"),
    #Input('editing-columns-button', "n_clicks"),
    Input('point-size', "value"),
    Input('editing-columns-name', 'value'),
    Input('nonZero', 'value'),
    )
    def update_graphs4(celltype,psize,value,zero):
        fig = []
        if 'ALL' in celltype:
            if 'Remove Zero' in zero:
                fig = px.scatter(df[df[value] != 0], x="UMAP1", y="UMAP2",color = 'Sample')
            else:
                fig = px.scatter(df, x="UMAP1", y="UMAP2",color = 'Sample')
        else:
            if 'Remove Zero' in zero:
                fig = px.scatter(df[df[value] != 0][df[df[value] != 0]['Cell Type'].isin(celltype)],
                                 x="UMAP1", y="UMAP2",color = 'Sample')
            else:
                fig = px.scatter(df[df['Cell Type'].isin(celltype)], x="UMAP1", y="UMAP2",color = 'Sample')
        fig.update_traces(marker=dict(size=psize))
        fig.update_traces(hoverinfo='none',hovertemplate=None)
        fig.update_layout(plot_bgcolor='rgba(0, 0, 0, 0)',)
        fig.update_layout(height = 600,width=699)
        return fig
    @app.callback(
    Output('indicator-graphic5', "figure"),
    Input('cell-type', "value"),
    #Input('editing-columns-button', "n_clicks"),
    Input('editing-columns-name', 'value'),
    Input('nonZero', 'value'),
    )
    def update_graphs5(celltype,value,zero):
        fig = []
        if 'ALL' in celltype:
            fig = []
        else:
            if 'Remove Zero' in zero:
                fig = px.violin(df[df[value] != 0][df[df[value] != 0]['Cell Type'].isin(celltype)], x= 'Cell Type',y=value,
                                color = 'Treatment',points="all")
            else:
                fig = px.violin(df[df['Cell Type'].isin(celltype)], x= 'Cell Type',y=value,
                                color = 'Treatment',points="all")
            fig.update_layout(plot_bgcolor='rgba(0, 0, 0, 0)',)
            fig.update_layout(height = 540,width=1620 )
        return fig
    context = {}
    return render(request, 'catalog/liver.html',context) 
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







#izize=20)=16,)Create your views here.
