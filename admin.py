from django.contrib import admin
from .models import Patient, Age, Gender, Status, Drug, Race, TAll,sampleid,experiment,MethylationProbe,DMP,DMR,lc50

admin.site.register(Patient)
admin.site.register(Age)
admin.site.register(Gender)
admin.site.register(Drug)
admin.site.register(Race)
admin.site.register(Status)
admin.site.register(TAll)
admin.site.register(sampleid)
admin.site.register(experiment)
admin.site.register(MethylationProbe)
admin.site.register(DMP)
admin.site.register(DMR)
admin.site.register(lc50)
# Register your models here.
