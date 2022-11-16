from django.db import models




class Age(models.Model):
        Age_Choices = (
        ('ADULT' , 'Adult'),
        ('PEDIATRIC' ,'Pediatric')
        )
        age = models.CharField(
        max_length = 20,
        choices = Age_Choices,
        )
        def __str__(self):
                """String for representing the Model object."""
                return self.age

class Gender(models.Model):
        Gender_Choices = (
        ('MALE' , 'Male'),
        ('FEMALE','Female')
        )
        gender = models.CharField(
        max_length = 20,
        choices = Gender_Choices,
        )
        def __str__(self):
                return self.gender

class Race(models.Model):
        race = models.CharField(max_length=10, help_text='Enter the race of the patient')

        def __str__(self):
                return self.race

class Status(models.Model):
        Dx = 'Dx'
        Ref ='Ref'
        Status_Choices = (
        ('Dx', 'Dx'),
        ('Ref' ,'Ref')
        )
        status = models.CharField(
        max_length = 10,
        choices = Status_Choices,
        )

        def __str__(self):
            """String for representing the MyModelName object (in Admin site etc.)."""
            return self.status

from django.urls import reverse

class Drug(models.Model):
        "Patient LC50 Information"""
        sampleid = models.CharField(primary_key=True, max_length=10, help_text='Enter a sample ID')
        patientid = models.ForeignKey('Patient',on_delete=models.SET_NULL, null=True)
        Dasatinib_LC50_nM = models.CharField(max_length=20, help_text='Enter a Dasatinib LC50 ')
        Venetoclax_LC50_nM = models.CharField(max_length=20, help_text='Enter a Venetoclax LC50 ')

        class Meta:
                ordering = ['sampleid']

        def __str__(self):
                return self.sampleid
        def get_absolute_url(self):
                return reverse('drug-detail', args=[str(self.sampleid)])


from django.urls import reverse


class Patient(models.Model):
        """Patient's General Information"""
        patientid = models.CharField(primary_key=True,max_length =13, help_text='Unique ID for this particular patient')
        gender = models.ForeignKey(Gender,on_delete=models.SET_NULL, null=True)
        age = models.ForeignKey(Age,on_delete=models.SET_NULL, null=True)
        status = models.ForeignKey(Status,on_delete=models.SET_NULL, null=True)
        race = models.ForeignKey(Race,on_delete=models.SET_NULL, null=True)

        class Meta:
                ordering = ['patientid']
        def __str__(self):
                return self.patientid

        def get_absolute_url(self):
        
                return reverse('patient-detail', args=[str(self.patientid)])





class TAll(models.Model):
        "Patient Detail Information"
        patientid = models.CharField(primary_key=True,max_length =10, help_text='Unique ID for this particular patient')
        age = models.ForeignKey(Age,on_delete=models.SET_NULL, null=True)
        status = models.ForeignKey(Status,on_delete=models.SET_NULL, null=True)
        dasatinib = models.CharField(max_length=10, default=0)
        venetoclax = models.CharField(max_length=10, default=0)
        SAMPLE_STATUS = (
            ('YES','YES'),
            ('NO','NO'),
            ('NULL','NA'),
        )
        fusion = models.CharField(max_length=15)
        SPI1_fusion_case = models.CharField(max_length=15,choices = SAMPLE_STATUS,blank = True,default='NULL')
        RNA_Seq = models.CharField(max_length=15,choices = SAMPLE_STATUS,blank = True,default='NULL')
        WES = models.CharField(max_length=5,choices = SAMPLE_STATUS,blank = True,default='NULL' )
        WGS = models.CharField(max_length=5,choices = SAMPLE_STATUS,blank = True,default='NULL' )
        SNP = models.CharField(max_length=5,choices = SAMPLE_STATUS,blank = True,default='NULL' )
        NetBID = models.CharField(max_length=5,choices = SAMPLE_STATUS,blank = True,default='NULL' )
        PDX = models.CharField(max_length=5,choices = SAMPLE_STATUS,blank = True,default='NULL' )
        scRNA = models.CharField(max_length=5,choices = SAMPLE_STATUS,blank = True,default='NULL' )

        class Meta:
            ordering = ['patientid']
        
        def __str__(self):
            return (self.patientid, self.age, self.status)
        def get_absolute_url(self):
                return reverse('patient-detail', args=[str(self.patientid)])


class experiment(models.Model):
        "Which experiment"
        exp = models.CharField(primary_key = True, max_length = 20)
        def __str__(self):
                return self.exp
        


class MethylationProbe(models.Model):
        "850K methylation probes"
        chromosome = models.CharField(max_length = 5)
        position = models.CharField(max_length=20)
        name = models.CharField(primary_key = True, max_length=35)
        gene = models.CharField(max_length=500,null = True ,blank = True,default='NULL' )
        class Meta:
                ordering = ['chromosome']
        def __str__(self):
                return self.name
        def get_absolute_url(self):
                return reverse('probes-detail', args=[str(self.name)])


class DMR(models.Model):
        "DMR Data"
        chromosome = models.CharField(max_length = 5)
        start = models.CharField(max_length=20)
        end = models.CharField(max_length=20)
        width = models.CharField(max_length=50)
        no_cpgs = models.CharField(max_length=15)
        min_smoothed_fdr = models.CharField(max_length=15)
        Stouffer = models.CharField(max_length=15)
        HMFDR = models.CharField(max_length=15)
        Fisher = models.CharField(max_length=15)
        maxdiff = models.CharField(max_length=15)
        meandiff = models.CharField(max_length=15)
        overlapping_genes = models.CharField(max_length=500,null = True,blank = True,default='NULL' )
        exp = models.ForeignKey(experiment,on_delete=models.SET_NULL, null=True)

        class Meta:
                ordering = ['Fisher']
        def __str__(self):
                return self.chromosome, self.start
        def get_absolute_url(self):
                return reverse('dmr-detail', args=[str(self.chromosome)])

class DMP(models.Model):
        "DMP Data"
        name = models.CharField(primary_key = True,max_length=30)
        chromosome = models.CharField(max_length = 5)
        position = models.CharField(max_length=30)
        gene = models.CharField(max_length=1000,null = True,blank = True,default='NULL' )
        logFC = models.CharField(max_length=15)
        AveExpr = models.CharField(max_length=15)
        t = models.CharField(max_length=15)
        pvalue = models.CharField(max_length=15)
        adjpvalue = models.CharField(max_length=15)
        B = models.CharField(max_length=15)
        exp = models.ForeignKey(experiment,on_delete=models.SET_NULL, null=True)

        class Meta:
                ordering = ['pvalue']
        def __str__(self):
                return self.name
        def get_absolute_url(self):
                return reverse('dmp-detail', args=[str(self.name)])

class Annotation850K(models.Model):
        "850K methylation annotation"
        chromosome = models.CharField(max_length = 5)
        position = models.CharField(max_length=20)
        name = models.ForeignKey(MethylationProbe,on_delete=models.SET_NULL, null=True)
        gene = models.CharField(max_length=255)

        class Meta:
                ordering = ['chromosome']
    
class sampleid(models.Model):
        "Smaple ID with Patient ID"
        pcgpid = models.CharField(primary_key = True, max_length = 30)
        id_type = models.CharField(max_length = 15)
        pharmgkbnumber = models.CharField(max_length = 50)
        institution = models.CharField(max_length = 30)
        
        def __str__(self):
                return self.pcgpid
        def get_absolute_url(self):
                return reverse('samples-detail', args=[str(self.pcgpid)])

class lc50(models.Model):
        age = models.CharField(max_length=50,null = True,blank = True,default='NULL' )
        age_group = models.CharField(max_length=50,null = True,blank = True,default='NULL' )
        gender = models.CharField(max_length=50,null = True,blank = True,default='NULL' )
        race = models.CharField(max_length=50,null = True,blank = True,default='NULL' )
        ethnicity = models.CharField(max_length=50,null = True,blank = True,default='NULL' )
        protocol = models.CharField(max_length=50,null = True,blank = True,default='NULL' )
        accession = models.CharField(max_length=50,null = True,blank = True,default='NULL' )
        assay_start_date = models.CharField(max_length=50,null = True,blank = True,default='NULL' )
        sample_date = models.CharField(max_length=50,null = True,blank = True,default='NULL' )
        date_of_birth = models.CharField(max_length=50,null = True,blank = True,default='NULL' )
        diseaseStatus = models.CharField(max_length=50,null = True,blank = True,default='NULL' )
        test_mnemonic = models.CharField(max_length=50,null = True,blank = True,default='NULL' )
        storage = models.CharField(max_length=50,null = True,blank = True,default='NULL' )
        blasts_qc = models.CharField(max_length=50,null = True,blank = True,default='NULL' )
        lineage = models.CharField(max_length=50,null = True,blank = True,default='NULL' )
        PercentBlasts = models.CharField(max_length=50,null = True,blank = True,default='NULL' )
        PercentBlastsPostFicoll = models.CharField(max_length=50,null = True,blank = True,default='NULL' )
        PercentBlastsDay4 = models.CharField(max_length=50,null = True,blank = True,default='NULL' )
        assay_type = models.CharField(max_length=50,null = True,blank = True,default='NULL' )
        pharmgkbnumber = models.CharField(max_length=50,null = True,blank = True,default='NULL' )
        drug = models.CharField(max_length=50,null = True,blank = True,default='NULL' )
        auc = models.CharField(max_length=50,null = True,blank = True,default='NULL' )
        rsquared = models.CharField(max_length=50,null = True,blank = True,default='NULL' )
        min_conc = models.CharField(max_length=50,null = True,blank = True,default='NULL' )
        max_conc = models.CharField(max_length=50,null = True,blank = True,default='NULL' )
        new_min = models.CharField(max_length=50,null = True,blank = True,default='NULL' )
        new_max = models.CharField(max_length=50,null = True,blank = True,default='NULL' )
        lc50 = models.CharField(max_length=50,null = True,blank = True,default='NULL' )
        new_lc50 = models.CharField(max_length=50,null = True,blank = True,default='NULL' )
        norm_form_lc50 = models.CharField(max_length=50,null = True,blank = True,default='NULL' )
        norm_lc50 = models.CharField(max_length=50,null = True,blank = True,default='NULL' )
        pcgpids = models.CharField(max_length=500,null = True,blank = True,default='NULL' )
        wentao_subtype = models.CharField(max_length=50,null = True,blank = True,default='NULL' )
        subtype = models.CharField(max_length=50,null = True,blank = True,default='NULL' )
        subtype_clean = models.CharField(max_length=50,null = True,blank = True,default='NULL' )

        def __str__(self):
                return self.pcgpids
        def get_absolute_url(self):
                return reverse('lc50-detail', args=[str(self.pcgpids)])


class discovery(models.Model):
        geneSymbol = models.CharField(max_length=50,null = True,blank = True,default='NULL' )
        funcType = models.CharField(max_length=50,null = True,blank = True,default='NULL' )
        size = models.IntegerField(null = True,blank = True,default='NULL')
        Z_Ped_Vs_Adult_DA = models.DecimalField(null = True,blank = True,default='NULL', max_digits=19, decimal_places=10)
        Z_Adult_sen_Vs_Adult_res_DA = models.DecimalField(null = True,blank = True,default='NULL', max_digits=19, decimal_places=10)
        Z_Ped_sen_Vs_Ped_res_DA = models.DecimalField(null = True,blank = True,default='NULL', max_digits=19, decimal_places=10)
        Z_sensitive_Vs_resistant_DA = models.DecimalField(null = True,blank = True,default='NULL', max_digits=19, decimal_places=10)
        AveExpr_Ped_Vs_Adult_DA = models.DecimalField(null = True,blank = True,default='NULL', max_digits=19, decimal_places=10)
        AveExpr_Adult_sen_Vs_Adult_res_DA = models.DecimalField(null = True,blank = True,default='NULL', max_digits=19, decimal_places=10)
        AveExpr_Ped_sen_Vs_Ped_res_DA = models.DecimalField(null = True,blank = True,default='NULL', max_digits=19, decimal_places=10)
        AveExpr_sensitive_Vs_resistant_DA = models.DecimalField(null = True,blank = True,default='NULL', max_digits=19, decimal_places=10)
        logFC_Ped_Vs_Adult_DA = models.DecimalField(null = True,blank = True,default='NULL', max_digits=19, decimal_places=10)
        logFC_Adult_sen_Vs_Adult_res_DA = models.DecimalField(null = True,blank = True,default='NULL', max_digits=19, decimal_places=10)
        logFC_Ped_sen_Vs_Ped_res_DA = models.DecimalField(null = True,blank = True,default='NULL', max_digits=19, decimal_places=10)
        logFC_sensitive_Vs_resistant_DA = models.DecimalField(null = True,blank = True,default='NULL', max_digits=19, decimal_places=10)
        P_Value_Ped_Vs_Adult_DA = models.DecimalField(null = True,blank = True,default='NULL', max_digits=19, decimal_places=10)
        P_Value_Adult_sen_Vs_Adult_res_DA = models.DecimalField(null = True,blank = True,default='NULL', max_digits=19, decimal_places=10)
        P_Value_Ped_sen_Vs_Ped_res_DA = models.DecimalField(null = True,blank = True,default='NULL', max_digits=19, decimal_places=10)
        P_Value_sensitive_Vs_resistant_DA = models.DecimalField(null = True,blank = True,default='NULL', max_digits=19, decimal_places=10)
        Z_Ped_Vs_Adult_DE = models.DecimalField(null = True,blank = True,default='NULL', max_digits=19, decimal_places=10)
        Z_Adult_sen_Vs_Adult_res_DE = models.DecimalField(null = True,blank = True,default='NULL', max_digits=19, decimal_places=10)
        Z_Ped_sen_Vs_Ped_res_DE = models.DecimalField(null = True,blank = True,default='NULL', max_digits=19, decimal_places=10)
        Z_sensitive_Vs_resistant_DE = models.DecimalField(null = True,blank = True,default='NULL', max_digits=19, decimal_places=10)
        AveExpr_Ped_Vs_Adult_DE = models.DecimalField(null = True,blank = True,default='NULL', max_digits=19, decimal_places=10)
        AveExpr_Adult_sen_Vs_Adult_res_DE = models.DecimalField(null = True,blank = True,default='NULL', max_digits=19, decimal_places=10)
        AveExpr_Ped_sen_Vs_Ped_res_DE = models.DecimalField(null = True,blank = True,default='NULL', max_digits=19, decimal_places=10)
        AveExpr_sensitive_Vs_resistant_DE = models.DecimalField(null = True,blank = True,default='NULL', max_digits=19, decimal_places=10)
        logFC_Ped_Vs_Adult_DE = models.DecimalField(null = True,blank = True,default='NULL', max_digits=19, decimal_places=10)
        logFC_Adult_sen_Vs_Adult_res_DE = models.DecimalField(null = True,blank = True,default='NULL', max_digits=19, decimal_places=10)
        logFC_Ped_sen_Vs_Ped_res_DE = models.DecimalField(null = True,blank = True,default='NULL', max_digits=19, decimal_places=10)
        logFC_sensitive_Vs_resistant_DE = models.DecimalField(null = True,blank = True,default='NULL', max_digits=19, decimal_places=10)
        P_Value_Ped_Vs_Adult_DE = models.DecimalField(null = True,blank = True,default='NULL', max_digits=19, decimal_places=10)
        P_Value_Adult_sen_Vs_Adult_res_DE = models.DecimalField(null = True,blank = True,default='NULL', max_digits=19, decimal_places=10)
        P_Value_Ped_sen_Vs_Ped_res_DE = models.DecimalField(null = True,blank = True,default='NULL', max_digits=19, decimal_places=10)
        P_Value_sensitive_Vs_resistant_DE = models.DecimalField(null = True,blank = True,default='NULL', max_digits=19, decimal_places=10)
        ensembl_gene_id = models.CharField(max_length=50,null = True,blank = True,default='NULL' )
        external_gene_name = models.CharField(max_length=50,null = True,blank = True,default='NULL' )
        gene_biotype = models.CharField(max_length=50,null = True,blank = True,default='NULL' )
        chromosome_name = models.CharField(max_length=50,null = True,blank = True,default='NULL' )
        strand = models.CharField(max_length=50,null = True,blank = True,default='NULL' )
        start_position = models.IntegerField(max_length=50,null = True,blank = True,default='NULL' )
        end_position = models.IntegerField(max_length=50,null = True,blank = True,default='NULL')

        def __str__(self):
                return self.geneSymbol
        def get_absolute_url(self):
                return reverse('discover-detail', args=[str(self.geneSymbol)])

class gene_drug(models.Model):
        name = models.CharField(primary_key = True, max_length = 30)
        def __str__(self):
                return self.name
class drug_groups(models.Model):
        treatment =  models.CharField(primary_key = True, max_length = 50)
        def __str__(self):
                return self.treatment
class drugs(models.Model):
        drugs = models.CharField(primary_key = True, max_length = 30)
        def __str__(self):
                return self.drugs

class screening(models.Model):
        name=models.CharField(max_length=50,null = True,blank = True,default='NULL' )
        Essential_lfc=models.DecimalField(null = True,blank = True,default='NULL', max_digits=10, decimal_places=10)
        Essential_T=models.DecimalField(null = True,blank = True,default='NULL', max_digits=10, decimal_places=10)
        Essential_pvalue=models.DecimalField(null = True,blank = True,default='NULL', max_digits=10, decimal_places=10)
        Essential_qvalue=models.DecimalField(null = True,blank = True,default='NULL', max_digits=10, decimal_places=10)
        Day6_Low_lfc=models.DecimalField(null = True,blank = True,default='NULL', max_digits=10, decimal_places=10)
        Day6_Low_T=models.DecimalField(null = True,blank = True,default='NULL', max_digits=10, decimal_places=10)
        Day6_Low_pvalue=models.DecimalField(null = True,blank = True,default='NULL', max_digits=10, decimal_places=10)
        Day6_Low_qvalue=models.DecimalField(null = True,blank = True,default='NULL', max_digits=10, decimal_places=10)
        Day6_High_lfc=models.DecimalField(null = True,blank = True,default='NULL', max_digits=10, decimal_places=10)
        Day6_High_T=models.DecimalField(null = True,blank = True,default='NULL', max_digits=10, decimal_places=10)
        Day6_High_pvalue=models.DecimalField(null = True,blank = True,default='NULL', max_digits=10, decimal_places=10)
        Day6_High_qvalue=models.DecimalField(null = True,blank = True,default='NULL', max_digits=10, decimal_places=10)
        Day12_Low_lfc=models.DecimalField(null = True,blank = True,default='NULL', max_digits=10, decimal_places=10)
        Day12_Low_T=models.DecimalField(null = True,blank = True,default='NULL', max_digits=10, decimal_places=10)
        Day12_Low_pvalue=models.DecimalField(null = True,blank = True,default='NULL', max_digits=10, decimal_places=10)
        Day12_Low_qvalue=models.DecimalField(null = True,blank = True,default='NULL', max_digits=10, decimal_places=10)
        Day12_High_lfc=models.DecimalField(null = True,blank = True,default='NULL', max_digits=10, decimal_places=10)
        Day12_High_T=models.DecimalField(null = True,blank = True,default='NULL', max_digits=10, decimal_places=10)
        Day12_High_pvalue=models.DecimalField(null = True,blank = True,default='NULL', max_digits=10, decimal_places=10)
        Day12_High_qvalue=models.DecimalField(null = True,blank = True,default='NULL', max_digits=10, decimal_places=10)
        Index=models.IntegerField(max_length=50,null = True,blank = True,default='NULL')
        drugs=models.CharField(max_length=50,null = True,blank = True,default='NULL' )
        
        def __str__(self):
               return self.name

class screen(models.Model):
        name=models.CharField(max_length=50,null = True,blank = True,default='NULL' )
        Essential_lfc=models.DecimalField(null = True,blank = True,default='NULL', max_digits=19, decimal_places=10)
        Essential_pvalue=models.DecimalField(null = True,blank = True,default='NULL', max_digits=19, decimal_places=10)
        Short_Low_lfc=models.DecimalField(null = True,blank = True,default='NULL', max_digits=19, decimal_places=10)
        Short_Low_pvalue=models.DecimalField(null = True,blank = True,default='NULL', max_digits=19, decimal_places=10)
        Short_High_lfc=models.DecimalField(null = True,blank = True,default='NULL', max_digits=19, decimal_places=10)
        Short_High_pvalue=models.DecimalField(null = True,blank = True,default='NULL', max_digits=19, decimal_places=10)
        Long_Low_lfc=models.DecimalField(null = True,blank = True,default='NULL', max_digits=19, decimal_places=10)
        Long_Low_pvalue=models.DecimalField(null = True,blank = True,default='NULL', max_digits=19, decimal_places=10)
        Long_High_lfc=models.DecimalField(null = True,blank = True,default='NULL', max_digits=19, decimal_places=10)
        Long_High_pvalue=models.DecimalField(null = True,blank = True,default='NULL', max_digits=19, decimal_places=10)
        Index=models.IntegerField(max_length=50,null = True,blank = True,default='NULL')
        drugs=models.CharField(max_length=50,null = True,blank = True,default='NULL' )
        CellLines =models.CharField(max_length=50,null = True,blank = True,default='NULL' )
        def __str__(self):
               return self.name

class lc50_detail(models.Model):
        age_group=models.CharField(max_length=50,null = True,blank = True,default='NULL' )
        diseaseStatus=models.CharField(max_length=50,null = True,blank = True,default='NULL' )
        lineage=models.CharField(max_length=50,null = True,blank = True,default='NULL' )
        assay_type=models.CharField(max_length=50,null = True,blank = True,default='NULL' )
        drug=models.CharField(max_length=50,null = True,blank = True,default='NULL' )
        status=models.CharField(max_length=50,null = True,blank = True,default='NULL' )
        subtype=models.CharField(max_length=50,null = True,blank = True,default='NULL' )
   
class sample_information (models.Model):
        groups =models.CharField(primary_key = True,max_length=50 )

        def __str__(self):
               return self.groups


   
