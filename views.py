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
                        if 'Boxplots' in boxplot:
                            if 'Violin' in violin:
                                if 'Dots' in dot:
                                    if group2 != 'N/A':
                                        fig = px.violin(dff,x = X, y = Y,points = 'all',color = groups,box=True) if groups != 'N/A' else px.violin(dff,x = X, y = Y,points = 'all',box=True)
                                    else:
                                        fig = px.violin(dff,x = X, y = Y,points = 'all',color = groups,box=True) if groups != 'N/A' else px.violin(dff,x = X, y = Y,points = 'all',box=True)
                                else:
                                    if group2 != 'N/A':
                                        fig = px.violin(dff,x = X,y=Y,points='outliers',color=groups,box=True) if groups != 'N/A' else px.violin(dff,x = X, y = Y,points = 'outliers',box=True)
                                    else:
                                        fig = px.violin(dff,x = X,y=Y,points='outliers',color=groups,box=True) if groups != 'N/A' else px.violin(dff,x = X, y = Y,points = 'outliers',box=True)
                            else:
                                if 'Dots' in dot:
                                    if group2 != 'N/A':
                                        fig = px.box(dff,x = X, y = Y,points = 'all',color = groups) if groups != 'N/A' else px.box(dff,x = X, y = Y,points = 'all')
                                    else:
                                        fig = px.box(dff,x = X, y = Y,points = 'all',color = groups) if groups != 'N/A' else px.box(dff,x = X, y = Y,points = 'all')
                                else:
                                    if group2 != 'N/A':
                                        fig = px.box(dff,x = X, y = Y,points = 'outliers',color = groups) if groups != 'N/A' else px.box(dff,x = X, y = Y,points = 'outliers')
                                    else:
                                        fig = px.box(dff,x = X, y = Y,points = 'outliers',color = groups) if groups != 'N/A' else px.box(dff,x = X, y = Y,points = 'outliers')
                        else:
                            if 'Violin' in violin:
                                if 'Dots' in dot:
                                    if group2 != 'N/A':
                                        fig = px.violin(dff,x = X, y = Y,points = 'all',color = groups) if groups != 'N/A' else px.violin(dff,x = X, y = Y,points = 'all')
                                    else:
                                        fig = px.violin(dff,x = X, y = Y,points = 'all',color = groups) if groups != 'N/A' else px.violin(dff,x = X, y = Y,points = 'all')
                                else:
                                    if group2 != 'N/A':
                                        fig = px.violin(dff,x=X, y = Y,points = 'outliers',color = groups) if groups != 'N/A' else px.violin(dff,x = X, y = Y,points = 'outliers')
                                    else:
                                        fig = px.violin(dff,x=X, y = Y,points = 'outliers',color = groups) if groups != 'N/A' else px.violin(dff,x = X, y = Y,points = 'outliers')
                            else:
                                if 'Dots' in dot:
                                    if  group2 != 'N/A':
                                        fig = px.strip(dff,x = X, y = Y,color = groups,symbol=group2) if groups != 'N/A' else px.strip(dff,x = X, y = Y,symbol=group2)
                                    else:
                                        fig = px.strip(dff,x = X, y = Y,color = groups) if groups != 'N/A' else px.strip(dff,x = X, y = Y)
                else:
                    if is_string_dtype(dff[Y]):
                        if 'Boxplots' in boxplot:
                            if 'Violin' in violin:
                                if 'Dots' in dot:
                                    if group2 != 'N/A':
                                        fig = px.violin(dff,x=X,y=Y,points='all',color=groups,box=True) if groups != 'N/A' else px.violin(dff,x = X, y = Y,points = 'all',box=True)
                                    else:
                                        fig = px.violin(dff,x=X,y=Y,points='all',color=groups,box=True) if groups != 'N/A' else px.violin(dff,x = X, y = Y,points = 'all',box=True)
                                else:
                                    if group2 != 'N/A':
                                        fig = px.violin(dff,x=X,y=Y,points='outliers',color=groups,box=True) if groups != 'N/A' else px.violin(dff,x=X,y=Y,points='outliers',box=True)
                                    else:
                                        fig = px.violin(dff,x=X,y=Y,points='outliers',color=groups,box=True) if groups != 'N/A' else px.violin(dff,x=X,y=Y,points='outliers',box=True)
                            else:
                                if 'Dots' in dot:
                                    if group2 != 'N/A':
                                        fig = px.box(dff,x = X, y = Y,points = 'all',color = groups) if groups != 'N/A' else px.box(dff,x = X, y = Y,points = 'all')
                                    else:
                                        fig = px.box(dff,x = X, y = Y,points = 'all',color = groups) if groups != 'N/A' else px.box(dff,x = X, y = Y,points = 'all')
                                else:
                                    if group2 != 'N/A':
                                        fig = px.box(dff,x = X, y = Y,points = 'outliers',color = groups) if groups != 'N/A' else px.box(dff,x = X, y = Y,points = 'outliers')
                                    else:
                                        fig = px.box(dff,x = X, y = Y,points = 'outliers',color = groups) if groups != 'N/A' else px.box(dff,x = X, y = Y,points = 'outliers')
                        else:
                            if 'Violin' in violin:
                                if 'Dots' in dot:
                                    if group2 != 'N/A':
                                        fig = px.violin(dff,x = X, y = Y,points = 'all',color = groups) if groups != 'N/A' else px.violin(dff,x = X, y = Y,points = 'all')
                                    else:
                                        fig = px.violin(dff,x = X, y = Y,points = 'all',color = groups) if groups != 'N/A' else px.violin(dff,x = X, y = Y,points = 'all')
                                else:
                                    if group2 != 'N/A':
                                        fig = px.violin(dff,x = X, y=Y,points = 'outliers',color = groups) if groups != 'N/A' else px.violin(dff,x = X, y = Y,points = 'outliers')
                                    else:
                                        fig = px.violin(dff,x = X, y = Y,points = 'outliers',color = groups) if groups != 'N/A' else px.violin(dff,x = X, y = Y,points = 'outliers')
                            else:
                                if 'Dots' in dot:
                                    if group2 != 'N/A':
                                        fig = px.strip(dff,x = X, y = Y,color = groups,symbol=group2) if groups != 'N/A' else px.strip(dff,x = X, y = Y,symbol=group2)
                                    else:
                                        fig = px.strip(dff,x = X, y = Y,color = groups) if groups != 'N/A' else px.strip(dff,x = X, y = Y)
                    else:
                        if groups != 'N/A':
                            if 'Trendlines' in trendline:
                                if 'Boxplots' in boxplot:
                                    if group2 != 'N/A':
                                        fig = px.scatter(dff, x=X, y=Y,trendline="ols",color = groups,symbol=group2,marginal_x ='box',marginal_y ='box',height = 800)
                                    else:
                                        fig = px.scatter(dff, x=X, y=Y,trendline="ols",color = groups,marginal_x ='box',marginal_y ='box',height = 800)
                                else:
                                    if group2 != 'N/A':
                                        fig = px.scatter(dff, x=X, y=Y,trendline="ols",color = groups,symbol=group2,height = 800)
                                    else:
                                        fig = px.scatter(dff, x=X, y=Y,trendline="ols",color = groups,height = 800)
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
                                        fig = px.scatter(dff, x=X, y=Y,trendline="ols",color=group2 ,marginal_x ='box',marginal_y ='box',height = 800)
                                    else:
                                        fig = px.scatter(dff, x=X, y=Y,trendline="ols" ,marginal_x ='box',marginal_y ='box',height = 800)
                                else:
                                    if group2 != 'N/A':
                                        fig = px.scatter(dff, x=X, y=Y,color=group2,trendline="ols",height = 800)
                                    else:
                                        fig = px.scatter(dff, x=X, y=Y,trendline="ols",height = 800)
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
    app.layout = html.Div([
        dcc.Upload(
            id='upload-data',
            children=html.Div([
                html.A('Select Files (xlsx,csv)')
            ]),
            style={
            'width': '30%',
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
        dcc.Upload(
            id='upload-image',
            children=html.Div([
                html.A('Select Image (jpg,png)')
            ]),
            style={
            'width': '30%',
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
        dcc.Graph(id='output-image-upload', clear_on_unhover=True,config={"displaylogo": False, 'modeBarButtonsToAdd':['zoom2d','drawopenpath','drawrect', 'eraseshape','resetViews','resetGeo'], }),
        html.Div(id="output-plot-upload"),
        html.Div(id='output-data-upload'),
        ])
    def parse_contents(contents, filename,date):
        content_type, content_string = contents.split(',')        
        decoded = base64.b64decode(content_string)
        try:
            if 'csv' in filename:
                df = pd.read_csv(
                    io.StringIO(decoded.decode('utf-8')))
                df['Status'] = ''
                df['color'] = ''
                if 'Hole Significance' in df.columns:
                    df['Hole Status'] = ''
                    df['Hole Status'] = df['Hole Significance'] * df['Hole Factor']
                else:
                    df['Hole Status'] = 999
                for i in range(len(df)):
                    if df.at[i,'Drug Interaction FOV - Number of Objects'] == 0:
                        df.at[i,'Status'] = 'Normal'
                        df.at[i,'color'] = '1'
                    elif df.at[i,'Peeling Factor'] < 0.0001:
                        df.at[i,'Status'] = 'Drug Interaction'
                        df.at[i,'color'] = '2'
                    elif df.at[i,'Empty Area Population - Drug Interaction FOV - CV % per Well'] < 9500:
                        if df.at[i,'Hole Status'] < 0.4:
                            df.at[i,'Status'] = 'Drug Interaction'
                            df.at[i,'color'] = '2'
                        else:
                            df.at[i,'Status'] = 'Peeling'
                            df.at[i,'color'] = '3'
                    elif df.at[i,'Empty Area Population - Drug Interaction FOV - CV % per Well'] > 9500:
                        if df.at[i,'Empty Ratio'] < 0.9:
                            if df.at[i,'Hole Status'] < 0.4:
                                df.at[i,'Status'] = 'Drug Interaction'
                                df.at[i,'color'] = '2'
                            else:
                                df.at[i,'Status'] = 'Peeling'
                                df.at[i,'color'] = '3'
                        else:
                            df.at[i,'Status'] = 'Drug Interaction'
                            df.at[i,'color'] = '2'
                    else:
                        df.at[i,'Status'] = 'Drug Interaction'
                        df.at[i,'color'] = '2'
            elif 'xls' in filename:
                df = pd.read_excel(io.BytesIO(decoded))
                df['Status'] = '' 
                df['color'] = ''
                if 'Hole Significance' in df.columns:
                    df['Hole Status'] = ''
                    df['Hole Status'] = df['Hole Significance'] * df['Hole Factor']
                else:
                    df['Hole Status'] = 999
                for i in range(len(df)):
                    if df.at[i,'Drug Interaction FOV - Number of Objects'] == 0:
                        df.at[i,'Status'] = 'Normal'
                        df.at[i,'color'] = '1'
                    elif df.at[i,'Peeling Factor'] < 0.0001:
                        df.at[i,'Status'] = 'Drug Interaction'
                        df.at[i,'color'] = '2'
                    elif df.at[i,'Empty Area Population - Drug Interaction FOV - CV % per Well'] < 9500:
                        if df.at[i,'Hole Status'] < 0.4:
                            df.at[i,'Status'] = 'Drug Interaction'
                            df.at[i,'color'] = '2'
                        else:
                            df.at[i,'Status'] = 'Peeling'
                            df.at[i,'color'] = '3'
                    elif df.at[i,'Empty Area Population - Drug Interaction FOV - CV % per Well'] > 9500:
                        if df.at[i,'Empty Ratio'] < 0.9:
                            if df.at[i,'Hole Status'] < 0.4:
                                df.at[i,'Status'] = 'Drug Interaction'
                                df.at[i,'color'] = '2'
                            else:
                                df.at[i,'Status'] = 'Peeling'
                                df.at[i,'color'] = '3'
                        else:
                            df.at[i,'Status'] = 'Drug Interaction'
                            df.at[i,'color'] = '2'
                    else:
                        df.at[i,'Status'] = 'Drug Interaction'
                        df.at[i,'color'] = '2'
            else:
                df = pd.read_csv(io.StringIO(decoded.decode('utf-8')),sep='\t',skiprows=9) 
                df.rename(columns={df.columns[-1]: 'Status'}, inplace=True)
                                       
        except Exception as e:
            print(e)
            return html.Div([
                'There was an error processing this file.'
                ])
        return html.Div([
            html.H5(filename),
            dash_table.DataTable(
                    df.to_dict('records'),
                    [{'name': i, 'id': i,"hideable":True} for i in df.columns],
                    page_size= 25,
                    filter_action="native",
                    filter_options = {'case':'insensitive'},
                    sort_action="native",
                    sort_mode="multi",
                    page_action="native",
                    #hidden_columns=['Display',"Use for Z'",'Plane','Timepoint','Normal FOV - Number of Objects','Height [µm]','Time [s]','Cell Type','color'],
                    export_format='xlsx',export_headers='display',merge_duplicate_headers=True,
                    style_table={'overflowX': 'auto'},
                    style_cell={'height': 'auto','overflow': 'hidden','font-family':'Helvetica'},style_data_conditional=[{'if': {'row_index': 'odd'},'backgroundColor': 'rgb(220,220,220)'}],
                    style_header={'backgroundColor': 'white','color': 'black','fontWeight': 'bold'},
                ),

                ])
    @app.callback(Output('output-data-upload', 'children'),
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'),
              State('upload-data','last_modified'))
    def update_output(list_of_contents, list_of_names, list_of_dates):
        if list_of_contents is not None:
            children = [
                parse_contents(c, n, d) for c, n, d in
                zip(list_of_contents, list_of_names, list_of_dates)]
            return children
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
            image = image_contents(contents)
            fig = go.Figure()
            if image is not None:
                fig = px.imshow(image,height = 738,width=1291.5,title = filenames)
                fig.update_xaxes(showticklabels=False)
                fig.update_yaxes(showticklabels=False)
                fig.add_annotation(x=(((image.shape[1] /21) * 1 ) - (image.shape[1] /42)), y=(image.shape[0] + (image.shape[0] /24)),text='3',showarrow=False,)
                fig.add_annotation(x=(((image.shape[1] /21) * 3 ) - (image.shape[1] /42)), y=(image.shape[0] + (image.shape[0] /24)),text='5',showarrow=False,)
                fig.add_annotation(x=(((image.shape[1] /21) * 8 ) - (image.shape[1] /42)), y=(image.shape[0] + (image.shape[0] /24)),text='10',showarrow=False,)
                fig.add_annotation(x=(((image.shape[1] /21) * 13 ) - (image.shape[1] /42)), y=(image.shape[0] + (image.shape[0] /24)),text='15',showarrow=False,)
                fig.add_annotation(x=(((image.shape[1] /21) * 18 ) - (image.shape[1] /42)), y=(image.shape[0] + (image.shape[0] /24)),text='20',showarrow=False,)
                fig.add_annotation(x=(0 - (image.shape[1] /42)), y=(((image.shape[0] /12) * 1 ) - (image.shape[0] /24)),text='B',showarrow=False,)
                fig.add_annotation(x=(0 - (image.shape[1] /42)), y=(((image.shape[0] /12) * 2 ) - (image.shape[0] /24)),text='C',showarrow=False,)
                fig.add_annotation(x=(0 - (image.shape[1] /42)), y=(((image.shape[0] /12) * 3 ) - (image.shape[0] /24)),text='D',showarrow=False,)
                fig.add_annotation(x=(0 - (image.shape[1] /42)), y=(((image.shape[0] /12) * 4 ) - (image.shape[0] /24)),text='E',showarrow=False,)
                fig.add_annotation(x=(0 - (image.shape[1] /42)), y=(((image.shape[0] /12) * 5 ) - (image.shape[0] /24)),text='F',showarrow=False,)
                fig.add_annotation(x=(0 - (image.shape[1] /42)), y=(((image.shape[0] /12) * 6 ) - (image.shape[0] /24)),text='G',showarrow=False,)
                fig.add_annotation(x=(0 - (image.shape[1] /42)), y=(((image.shape[0] /12) * 7 ) - (image.shape[0] /24)),text='J',showarrow=False,)
                fig.add_annotation(x=(0 - (image.shape[1] /42)), y=(((image.shape[0] /12) * 8 ) - (image.shape[0] /24)),text='K',showarrow=False,)
                fig.add_annotation(x=(0 - (image.shape[1] /42)), y=(((image.shape[0] /12) * 9 ) - (image.shape[0] /24)),text='L',showarrow=False,)
                fig.add_annotation(x=(0 - (image.shape[1] /42)), y=(((image.shape[0] /12) * 10 ) - (image.shape[0] /24)),text='M',showarrow=False,)
                fig.add_annotation(x=(0 - (image.shape[1] /42)), y=(((image.shape[0] /12) * 11 ) - (image.shape[0] /24)),text='N',showarrow=False,)
                fig.add_annotation(x=(0 - (image.shape[1] /42)), y=(((image.shape[0] /12) * 12 ) - (image.shape[0] /24)),text='O',showarrow=False,)
                fig.add_shape(type="rect",x0=0,y0=0,x1=((image.shape[1] /21) * 1) ,y1=image.shape[0] /2,line=dict(color="Grey",width=3,))
                fig.add_shape(type="rect",x0=((image.shape[1] /21) * 1),y0=0,x1=((image.shape[1] /21) * 3) ,y1=image.shape[0] /2,line=dict(color="Grey",width=3,))
                fig.add_shape(type="rect",x0=((image.shape[1] /21) * 3),y0=0,x1=((image.shape[1] /21) * 5) ,y1=image.shape[0] /2,line=dict(color="Grey",width=3,))
                fig.add_shape(type="rect",x0=((image.shape[1] /21) * 5),y0=0,x1=((image.shape[1] /21) * 7) ,y1=image.shape[0] /2,line=dict(color="Grey",width=3,))
                fig.add_shape(type="rect",x0=((image.shape[1] /21) * 7),y0=0,x1=((image.shape[1] /21) * 9) ,y1=image.shape[0] /2,line=dict(color="Grey",width=3,))
                fig.add_shape(type="rect",x0=((image.shape[1] /21) * 9),y0=0,x1=((image.shape[1] /21) * 11) ,y1=image.shape[0] /2,line=dict(color="Grey",width=3,))
                fig.add_shape(type="rect",x0=((image.shape[1] /21) * 11),y0=0,x1=((image.shape[1] /21) * 13) ,y1=image.shape[0] /2,line=dict(color="Grey",width=3,))
                fig.add_shape(type="rect",x0=((image.shape[1] /21) * 13),y0=0,x1=((image.shape[1] /21) * 15) ,y1=image.shape[0] /2,line=dict(color="Grey",width=3,))
                fig.add_shape(type="rect",x0=((image.shape[1] /21) * 15),y0=0,x1=((image.shape[1] /21) * 17) ,y1=image.shape[0] /2,line=dict(color="Grey",width=3,))
                fig.add_shape(type="rect",x0=((image.shape[1] /21) * 17),y0=0,x1=((image.shape[1] /21) * 19) ,y1=image.shape[0] /2,line=dict(color="Grey",width=3,))
                fig.add_shape(type="rect",x0=((image.shape[1] /21) * 19),y0=0,x1=((image.shape[1] /21) * 21) ,y1=image.shape[0] /2,line=dict(color="Grey",width=3,))
                fig.add_shape(type="rect",x0=0,y0=image.shape[0] /2,x1=((image.shape[1] /21) * 1) ,y1=image.shape[0],line=dict(color="Grey",width=3,))
                fig.add_shape(type="rect",x0=((image.shape[1] /21) * 1),y0=image.shape[0] /2,x1=((image.shape[1] /21) * 3) ,y1=image.shape[0],line=dict(color="Grey",width=3,))
                fig.add_shape(type="rect",x0=((image.shape[1] /21) * 3),y0=image.shape[0] /2,x1=((image.shape[1] /21) * 5) ,y1=image.shape[0],line=dict(color="Grey",width=3,))
                fig.add_shape(type="rect",x0=((image.shape[1] /21) * 5),y0=image.shape[0] /2,x1=((image.shape[1] /21) * 7) ,y1=image.shape[0],line=dict(color="Grey",width=3,))
                fig.add_shape(type="rect",x0=((image.shape[1] /21) * 7),y0=image.shape[0] /2,x1=((image.shape[1] /21) * 9) ,y1=image.shape[0],line=dict(color="Grey",width=3,))
                fig.add_shape(type="rect",x0=((image.shape[1] /21) * 9),y0=image.shape[0] /2,x1=((image.shape[1] /21) * 11) ,y1=image.shape[0],line=dict(color="Grey",width=3,))
                fig.add_shape(type="rect",x0=((image.shape[1] /21) * 11),y0=image.shape[0] /2,x1=((image.shape[1] /21) * 13) ,y1=image.shape[0],line=dict(color="Grey",width=3,))
                fig.add_shape(type="rect",x0=((image.shape[1] /21) * 13),y0=image.shape[0] /2,x1=((image.shape[1] /21) * 15) ,y1=image.shape[0],line=dict(color="Grey",width=3,))
                fig.add_shape(type="rect",x0=((image.shape[1] /21) * 15),y0=image.shape[0] /2,x1=((image.shape[1] /21) * 17) ,y1=image.shape[0],line=dict(color="Grey",width=3,))
                fig.add_shape(type="rect",x0=((image.shape[1] /21) * 17),y0=image.shape[0] /2,x1=((image.shape[1] /21) * 19) ,y1=image.shape[0],line=dict(color="Grey",width=3,))
                fig.add_shape(type="rect",x0=((image.shape[1] /21) * 19),y0=image.shape[0] /2,x1=((image.shape[1] /21) * 21) ,y1=image.shape[0],line=dict(color="Grey",width=3,))
                fig.update_traces(hoverinfo='none',hovertemplate=None)
                fig.update_layout(plot_bgcolor='rgba(0, 0, 0, 0)',)
                return fig
        return {'data': [], 'layout': go.Layout()}
        
        #if list_of_contents is not None:
         #   children = [
          #      image_contents(c, n) for c, n,  in
           #     zip(list_of_contents, list_of_names)]
            #return children
    def plot_contents(table_contents, table_filename, table_date,img_contents,img_filename,img_date ):
        table_content_type, table_content_string = table_contents.split(',')        
        decoded = base64.b64decode(table_content_string)
        if 'csv' in table_filename:
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            df['Status'] = ''
            df['color'] = ''
            if 'Hole Significance' in df.columns:
                    df['Hole Status'] = ''
                    df['Hole Status'] = df['Hole Significance'] * df['Hole Factor']
            else:
                    df['Hole Status'] = 999
            for i in range(len(df)):
                if df.at[i,'Drug Interaction FOV - Number of Objects'] == 0:
                    df.at[i,'Status'] = 'Normal'
                    df.at[i,'color'] = '1'
                elif df.at[i,'Peeling Factor'] < 0.0001:
                    df.at[i,'Status'] = 'Drug Interaction'
                    df.at[i,'color'] = '2'
                elif df.at[i,'Empty Area Population - Drug Interaction FOV - CV % per Well'] < 9500:
                    if df.at[i,'Hole Status'] < 0.4:
                        df.at[i,'Status'] = 'Drug Interaction'
                        df.at[i,'color'] = '2'
                    else:
                        df.at[i,'Status'] = 'Peeling'
                        df.at[i,'color'] = '3'
                df.at[i,'Status'] = 'Drug Interaction'elif df.at[i,'Empty Area Population - Drug Interaction FOV - CV % per Well'] > 9500:
                    if df.at[i,'Empty Ratio'] < 0.9:
                        if df.at[i,'Hole Status'] < 0.4:
                            df.at[i,'Status'] = 'Drug Interaction'
                            df.at[i,'color'] = '2'
                        else:
                            df.at[i,'Status'] = 'Peeling'
                            df.at[i,'color'] = '3'
                    else:
                        df.at[i,'Status'] = 'Drug Interaction'
                        df.at[i,'color'] = '2'
                else:
                    df.at[i,'Status'] = 'Drug Interaction'
                    df.at[i,'color'] = '2'
        elif 'xls' in table_filename:
            df = pd.read_excel(io.BytesIO(decoded))
            df['Status'] = ''
            df['color'] = ''
            if 'Hole Significance' in df.columns:
                    df['Hole Status'] = ''
                    df['Hole Status'] = df['Hole Significance'] * df['Hole Factor']
            else:
                    df['Hole Status'] = 999
            for i in range(len(df)):
                if df.at[i,'Drug Interaction FOV - Number of Objects'] == 0:
                    df.at[i,'Status'] = 'Normal'
                    df.at[i,'color'] = '1'
                elif df.at[i,'Peeling Factor'] < 0.0001:
                    df.at[i,'Status'] = 'Drug Interaction'
                    df.at[i,'color'] = '2'
                elif df.at[i,'Empty Area Population - Drug Interaction FOV - CV % per Well'] < 9500:
                    if df.at[i,'Hole Status'] < 0.4:
                        df.at[i,'Status'] = 'Drug Interaction'
                        df.at[i,'color'] = '2'
                    else:
                        df.at[i,'Status'] = 'Peeling'
                        df.at[i,'color'] = '3'
                elif df.at[i,'Empty Area Population - Drug Interaction FOV - CV % per Well'] > 9500:
                    if df.at[i,'Empty Ratio'] < 0.9:
                        if df.at[i,'Hole Status'] < 0.4:
                            df.at[i,'Status'] = 'Drug Interaction'
                            df.at[i,'color'] = '2'
                        else:
                            df.at[i,'Status'] = 'Peeling'
                            df.at[i,'color'] = '3'
                    else:
                        df.at[i,'Status'] = 'Drug Interaction'
                        df.at[i,'color'] = '2'
                else:
                    df.at[i,'Status'] = 'Drug Interaction'
                    df.at[i,'color'] = '2'
        fig = px.imshow(df.pivot('Row', 'Column', 'color'),color_continuous_scale='Blues')
        fig.update(data=[{'customdata': df.pivot('Row', 'Column', 'Status'),'hovertemplate': 'Coloum: %{x}<br>Row: %{y}<br>Status: %{customdata}<br><extra></extra>'}])
        fig.update_traces(dict(showscale=False, coloraxis=None,colorscale='Blues'))
        fig.update_layout(height=738,width=1291.5,)
        fig.update_layout(plot_bgcolor='rgba(0, 0, 0, 0)',)
        fig.add_vline(x=3.5,line_width=1, line_dash="dash"),fig.add_shape(x0=4.5,x1 = 4.5, y0=-0.5,y1 = 11.5 ,line_width=1, line_dash="dash"),fig.add_vline(x=5.5,line_width=1, line_dash="dash")
        fig.add_shape(x0=6.5,x1 = 6.5,y0 = -0.5,y1= 11.5,line_width=1, line_dash="dash"),fig.add_vline(x=7.5,line_width=1, line_dash="dash")
        fig.add_shape(x0=8.5,x1 = 8.5,y0=-0.5,y1 = 11.5,line_width=1, line_dash="dash")
        fig.add_vline(x=9.5,line_width=1, line_dash="dash")
        fig.add_shape(x0=10.5,x1 = 10.5,y0=-0.5,y1 = 11.5,line_width=1, line_dash="dash"),fig.add_vline(x=11.5,line_width=1, line_dash="dash")
        fig.add_shape(x0=12.5,x1=12.5,y0=-0.5,y1 = 11.5,line_width=1, line_dash="dash")
        fig.add_shape(x0=2.5,x1 = 2.5,y0=-0.5,y1 = 11.5,line_width=1, line_dash="dash"),fig.add_vline(x=13.5,line_width=1, line_dash="dash")
        fig.add_shape(x0=14.5,x1 = 14.5,y0=-0.5,y1 = 11.5,line_width=1, line_dash="dash"),
        fig.add_vline(x=15.5,line_width=1, line_dash="dash"),fig.add_shape(x0=16.5,x1 = 16.5,y0=-0.5,y1 = 11.5,line_width=1, line_dash="dash"),fig.add_vline(x=17.5,line_width=1, line_dash="dash")
        fig.add_shape(x0=18.5,x1 = 18.5,y0=-0.5,y1 = 11.5,line_width=1, line_dash="dash"),fig.add_vline(x=19.5,line_width=1, line_dash="dash")
        fig.add_shape(x0=20.5,x1 = 20.5,y0=-0.5,y1 = 11.5,line_width=1, line_dash="dash")
        fig.add_vline(x=21.5,line_width=1, line_dash="dash"),fig.add_shape(x0=22.5,x1 = 22.5,y0=-0.5,y1 = 11.5,line_width=1, line_dash="dash")
        fig.add_hline(y=1.5,line_width=1, line_dash="dash"),fig.add_hline(y=2.5,line_width=1, line_dash="dash"),fig.add_hline(y=3.5,line_width=1, line_dash="dash")
        fig.add_hline(y=4.5,line_width=1, line_dash="dash"),fig.add_hline(y=5.5,line_width=1, line_dash="dash"),fig.add_hline(y=6.5,line_width=1, line_dash="dash")
        fig.add_hline(y=7.5,line_width=1, line_dash="dash"),fig.add_hline(y=8.5,line_width=1, line_dash="dash"),
        fig.add_hline(y=9.5,line_width=1, line_dash="dash"),fig.add_hline(y=0.5,line_width=1, line_dash="dash"),fig.add_hline(y=10.5,line_width=1, line_dash="dash")
        fig.add_shape(type="rect",x0=2.5, y0=3.5, x1=3.5, y1=4.5,fillcolor="Grey",),fig.add_shape(type="rect",x0=2.5, y0=9.5, x1=3.5, y1=10.5,fillcolor="Grey",)
        fig.add_shape(type="rect",x0=2.5,y0=-0.5,x1=3.5,y1=5.5)
        fig.add_annotation(x=3, y=-1.0,text='Control',showarrow=False,)
        fig.add_shape(type="rect",x0=3.5,y0=-0.5,x1=5.5,y1=5.5)
        fig.add_annotation(x=4.5, y=-1.0,text=df.loc[(df['Row'] == 'B') & (df['Column'] == 4), 'Compound'].values[0],showarrow=False,)
        fig.add_shape(type="rect",x0=5.5,y0=-0.5,x1=7.5,y1=5.5)
        fig.add_annotation(x=6.5, y=-1.0,text=df.loc[(df['Row'] == 'B') & (df['Column'] == 6), 'Compound'].values[0],showarrow=False,)
        fig.add_shape(type="rect",x0=7.5,y0=-0.5,x1=9.5,y1=5.5)
        fig.add_annotation(x=8.5, y=-1.0,text=df.loc[(df['Row'] == 'B') & (df['Column'] == 8), 'Compound'].values[0],showarrow=False,)
        fig.add_shape(type="rect",x0=9.5,y0=-0.5,x1=11.5,y1=5.5)
        fig.add_annotation(x=10.5, y=-1.0,text=df.loc[(df['Row'] == 'B') & (df['Column'] == 10), 'Compound'].values[0],showarrow=False,)
        fig.add_shape(type="rect",x0=11.5,y0=-0.5,x1=13.5,y1=5.5)
        fig.add_annotation(x=12.5, y=-1.0,text=df.loc[(df['Row'] == 'B') & (df['Column'] == 12), 'Compound'].values[0],showarrow=False,)
        fig.add_shape(type="rect",x0=13.5,y0=-0.5,x1=15.5,y1=5.5)
        fig.add_annotation(x=14.5, y=-1.0,text=df.loc[(df['Row'] == 'B') & (df['Column'] == 14), 'Compound'].values[0],showarrow=False,)
        fig.add_shape(type="rect",x0=15.5,y0=-0.5,x1=17.5,y1=5.5)
        fig.add_annotation(x=16.5, y=-1.0,text=df.loc[(df['Row'] == 'B') & (df['Column'] == 16), 'Compound'].values[0],showarrow=False,)
        fig.add_shape(type="rect",x0=17.5,y0=-0.5,x1=19.5,y1=5.5)
        fig.add_annotation(x=18.5, y=-1.0,text=df.loc[(df['Row'] == 'B') & (df['Column'] == 18), 'Compound'].values[0],showarrow=False,)
        fig.add_shape(type="rect",x0=19.5,y0=-0.5,x1=21.5,y1=5.5)
        fig.add_annotation(x=20.5, y=-1.0,text=df.loc[(df['Row'] == 'B') & (df['Column'] == 20), 'Compound'].values[0],showarrow=False,)
        fig.add_shape(type="rect",x0=21.5,y0=-0.5,x1=23.5,y1=5.5)
        fig.add_annotation(x=22.5, y=-1.0,text=df.loc[(df['Row'] == 'B') & (df['Column'] == 22), 'Compound'].values[0],showarrow=False,) 
        fig.add_shape(type="rect",x0=2.5,y0=5.5,x1=3.5,y1=11.5)
        fig.add_annotation(x=3, y=12.0,text='Control',showarrow=False,)
        fig.add_shape(type="rect",x0=3.5,y0=5.5,x1=5.5,y1=11.5) 
        fig.add_annotation(x=4.5, y=12.0,text=df.loc[(df['Row'] == 'J') & (df['Column'] == 4), 'Compound'].values[0],showarrow=False,)
        fig.add_shape(type="rect",x0=5.5,y0=5.5,x1=7.5,y1=11.5)
        fig.add_annotation(x=6.5, y=12.0,text=df.loc[(df['Row'] == 'J') & (df['Column'] == 6), 'Compound'].values[0],showarrow=False,)
        fig.add_shape(type="rect",x0=7.5,y0=5.5,x1=9.5,y1=11.5)
        fig.add_annotation(x=8.5, y=12.0,text=df.loc[(df['Row'] == 'J') & (df['Column'] == 8), 'Compound'].values[0],showarrow=False,)
        fig.add_shape(type="rect",x0=9.5,y0=5.5,x1=11.5,y1=11.5)
        fig.add_annotation(x=10.5, y=12.0,text=df.loc[(df['Row'] == 'J') & (df['Column'] == 10), 'Compound'].values[0],showarrow=False,)
        fig.add_shape(type="rect",x0=11.5,y0=5.5,x1=13.5,y1=11.5)
        fig.add_annotation(x=12.5, y=12.0,text=df.loc[(df['Row'] == 'J') & (df['Column'] == 12), 'Compound'].values[0],showarrow=False,)
        fig.add_shape(type="rect",x0=13.5,y0=5.5,x1=15.5,y1=11.5)
        fig.add_annotation(x=14.5, y=12.0,text=df.loc[(df['Row'] == 'J') & (df['Column'] == 14), 'Compound'].values[0],showarrow=False,)
        fig.add_shape(type="rect",x0=15.5,y0=5.5,x1=17.5,y1=11.5)
        fig.add_annotation(x=16.5, y=12.0,text=df.loc[(df['Row'] == 'J') & (df['Column'] == 16), 'Compound'].values[0],showarrow=False,)
        fig.add_shape(type="rect",x0=17.5,y0=5.5,x1=19.5,y1=11.5)
        fig.add_annotation(x=18.5, y=12.0,text=df.loc[(df['Row'] == 'J') & (df['Column'] == 18), 'Compound'].values[0],showarrow=False,)
        fig.add_shape(type="rect",x0=19.5,y0=5.5,x1=21.5,y1=11.5)
        fig.add_annotation(x=20.5, y=12.0,text=df.loc[(df['Row'] == 'J') & (df['Column'] == 20), 'Compound'].values[0],showarrow=False,)
        fig.add_shape(type="rect",x0=21.5,y0=5.5,x1=23.5,y1=11.5)
        fig.add_annotation(x=22.5, y=12.0,text=df.loc[(df['Row'] == 'J') & (df['Column'] == 22), 'Compound'].values[0],showarrow=False,)
        fig.add_shape(type="rect",x0=2.5,y0=5.5,x1=3.5,y1=11.5)
        return html.Div([
            dcc.Graph(id="indicator-graphic", figure=fig, clear_on_unhover=True,config={"displaylogo": False}),
            ])
    @app.callback(
        Output("output-plot-upload", 'children'),
        Input('upload-data', 'contents'),
        Input('upload-image', 'contents'),
        State('upload-data', 'filename'),
        State('upload-data', 'last_modified'),
        State('upload-image', 'filename'),
        State('upload-image', 'last_modified'),
        )
    def plot_graph(table_contents,img_contents, table_names, table_dates, img_filename,img_last_modified):
        if table_contents is not None:
            if img_contents is not None:
                children = [plot_contents(c, n, d, cc, nn, dd) for c, n, d, cc, nn, dd in 
                    zip(table_contents, table_names, table_dates,img_contents,img_filename,img_last_modified)]
                return children

    context = {}
    return render(request, 'catalog/image.html',context)    


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
