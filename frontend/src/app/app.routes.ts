import { Routes } from '@angular/router';
import { HomeComponent } from './components/home/home-component/home-component';
import { SettingsComponent } from './components/settings/settings-component/settings-component';
import { DatasetsComponent } from './components/datasets/datasets-component/datasets-component';
import { ModelsComponent } from './components/models/models-component/models-component';
import { MaskGenerationComponent } from './components/mask-generation/mask-generation-component/mask-generation-component';
import { HeatmapGenerationComponent } from './components/heatmap-generation/heatmap-generation-component/heatmap-generation-component';
import { UnsupervisedActivations } from './components/unsupervised-activations/unsupervised-activations';

export const routes: Routes = [
    {
        path: 'home', component: HomeComponent, pathMatch: 'full'
    },
    {
        path: 'settings', component: SettingsComponent, pathMatch: 'full'
    },
    {
        path: 'datasets', component: DatasetsComponent, pathMatch: 'full'
    },
    {
        path: 'models', component: ModelsComponent, pathMatch: 'full'
    },
    {
        path: 'mask-generation', component: MaskGenerationComponent, pathMatch: 'full'
    },
    {
        path: 'heatmap-generation', component: HeatmapGenerationComponent, pathMatch: 'full'
    },
    {
        path: 'unsupervised-activations', component: UnsupervisedActivations, pathMatch: 'full'
    },
    {
        path: '**', redirectTo: 'home', pathMatch: 'full'
    }
];
