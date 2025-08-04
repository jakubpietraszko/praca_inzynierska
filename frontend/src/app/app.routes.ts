import { Routes } from '@angular/router';
import { HomeComponent } from './components/home/home-component/home-component';
import { SettingsComponent } from './components/settings/settings-component/settings-component';
import { DatasetsComponent } from './components/datasets/datasets-component/datasets-component';
import { ModelsComponent } from './components/models/models-component/models-component';

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
        path: '**', redirectTo: 'home', pathMatch: 'full'
    }
];
