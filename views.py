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
                'groups',
                dcc.Dropdown(plot_options,value = 'N/A',id='groups')],style={'width': '25%', 'display': 'inline-block'}),
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
    Output("groups", "options"),
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
    [Input('x-axis', 'value'),Input('y-axis', 'value'),Input('groups','value'),Input('trendline','value'),Input('boxplot','value'),
    Input('violin','value'),Input('dot','value'),Input('Reverse','value'),Input('slider-updatemode','value') ],
    Input('datatable-interactivity', "derived_virtual_data"),
    )
    def update_graphs(X,Y,groups,trendline,boxplot,violin,dot,reverse,slider,data):
        dff = pd.DataFrame(data)
        if 'Reverse' in reverse:
            XX = X
            X = Y
            Y = XX
        if X:
            if Y:
                dff = dff.dropna(subset=[X,Y,groups]) if groups != 'N/A' else dff.dropna(subset=[X,Y]) 
                if is_string_dtype(dff[X]):
                    if is_string_dtype(dff[Y]):
                        fig = px.histogram(dff, x=X, color = Y,text_auto=True,height = 800).update_xaxes(categoryorder='total descending')
                    else:
                        if 'Boxplots' in boxplot:
                            if 'Violin' in violin:
                                if 'Dots' in dot:
                                    fig = px.violin(dff,x = X, y = Y,points = 'all',color = groups,box=True) if groups != 'N/A' else px.violin(dff,x = X, y = Y,points = 'all',box=True)
                                else:
                                    fig = px.violin(dff,x = X, y = Y,points = 'outliers',color = groups,box=True) if groups != 'N/A' else px.violin(dff,x = X, y = Y,points = 'outliers',box=True)
                            else:
                                if 'Dots' in dot:
                                    fig = px.box(dff,x = X, y = Y,points = 'all',color = groups) if groups != 'N/A' else px.box(dff,x = X, y = Y,points = 'all')
                                else:
                                    fig = px.box(dff,x = X, y = Y,points = 'outliers',color = groups) if groups != 'N/A' else px.box(dff,x = X, y = Y,points = 'outliers')
                        else:
                            if 'Violin' in violin:
                                if 'Dots' in dot:
                                    fig = px.violin(dff,x = X, y = Y,points = 'all',color = groups) if groups != 'N/A' else px.violin(dff,x = X, y = Y,points = 'all')
                                else:
                                    fig = px.violin(dff,x = X, y = Y,points = 'outliers',color = groups) if groups != 'N/A' else px.violin(dff,x = X, y = Y,points = 'outliers')
                            else:
                                if 'Dots' in dot:
                                    fig = px.strip(dff,x = X, y = Y,color = groups) if groups != 'N/A' else px.strip(dff,x = X, y = Y)
                else:
                    if is_string_dtype(dff[Y]):
                        if 'Boxplots' in boxplot:
                            if 'Violin' in violin:
                                if 'Dots' in dot:
                                    fig = px.violin(dff,x = X, y = Y,points = 'all',color = groups,box=True) if groups != 'N/A' else px.violin(dff,x = X, y = Y,points = 'all',box=True)
                                else:
                                    fig = px.violin(dff,x = X, y = Y,points = 'outliers',color = groups,box=True) if groups != 'N/A' else px.violin(dff,x = X, y = Y,points = 'outliers',box=True)
                            else:
                                if 'Dots' in dot:
                                    fig = px.box(dff,x = X, y = Y,points = 'all',color = groups) if groups != 'N/A' else px.box(dff,x = X, y = Y,points = 'all')
                                else:
                                    fig = px.box(dff,x = X, y = Y,points = 'outliers',color = groups) if groups != 'N/A' else px.box(dff,x = X, y = Y,points = 'outliers')
                        else:
                            if 'Violin' in violin:
                                if 'Dots' in dot:
                                    fig = px.violin(dff,x = X, y = Y,points = 'all',color = groups) if groups != 'N/A' else px.violin(dff,x = X, y = Y,points = 'all')
                                else:
                                    fig = px.violin(dff,x = X, y = Y,points = 'outliers',color = groups) if groups != 'N/A' else px.violin(dff,x = X, y = Y,points = 'outliers')
                            else:
                                if 'Dots' in dot:
                                    fig = px.strip(dff,x = X, y = Y,color = groups) if groups != 'N/A' else px.strip(dff,x = X, y = Y)

                    else:
                        if groups != 'N/A':
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
    def temp():
        return
    context = {}
    return render(request, 'catalog/segment.html',context)
@login_required(login_url='/accounts/login/')
def crispr(request):
    mp = pd.read_csv('static/CRISPR_6MP_REH.csv')
    app = DjangoDash('app_crispr',external_stylesheets=[dbc.themes.BOOTSTRAP],add_bootstrap_links=True)
    app.layout = html.Div([
        dcc.Input(id='filter-query-input', placeholder='Enter filter query  "Example: {TP53}>8 and {LCK}<12 and {drug} contains DEX and {column_name}=... "',debounce=True,style={'width':'1000px'} ),
        html.Div([dcc.Dropdown(['6-MP','AraC','Daunorubicin','L-asparaginase','Maphosphamide','Methotrexate','Vincristine','Trametinib','Dasatinib'],'6-MP',id='drug-name',style={'width':'50%'})]),
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
            df = pd.read_csv('static/CRISPR_Methotrexate_REH.csv')
        elif drug == 'Trametinib':
            df = pd.read_parquet('static/Web_Screening.parquet')
            df = df[df['Drugs'].str.contains('Trametinib')]
        elif drug == 'Dasatinib':
            df = pd.read_parquet('static/Web_Screening.parquet')
            df = df[df['Drugs'].str.contains('Dasatinib')]
        else:
            df = pd.read_csv('static/CRISPR_Vincristine_REH.csv')
        columns = [
            {"name": i, "id": i, "deletable": False, "selectable": True,"hideable":True} for i in df.columns
            ]
        data = df.to_dict('records')
        return data,columns
    context = {}
    return render(request, 'catalog/crispr.html',context)



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
