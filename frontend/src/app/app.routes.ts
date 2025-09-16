import { Routes } from '@angular/router';
import { HomeComponent } from './components/home/home-component/home-component';
import { SettingsComponent } from './components/settings/settings-component/settings-component';
import { DatasetsComponent } from './components/datasets/datasets-component/datasets-component';
import { ModelsComponent } from './components/models/models-component/models-component';
import { UnsupervisedActivations } from './components/unsupervised-activations/unsupervised-activations';
import { UnsupervisedActivationsWrapper } from './components/unsupervised-activations-wrapper/unsupervised-activations-wrapper';

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
        path: 'unsupervised-activations', component: UnsupervisedActivationsWrapper, pathMatch: 'full'
    },
    {
        path: '**', redirectTo: 'home', pathMatch: 'full'
    }
];
