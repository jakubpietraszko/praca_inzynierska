import { Routes } from '@angular/router';
import { SettingsComponent } from './components/settings/settings-component/settings-component';
import { DatasetsComponent } from './components/datasets/datasets-component/datasets-component';
import { Generator } from './components/generator-page/generator';
import { Home } from './components/home-page/home/home';
import { Statistics } from './components/statistics-page/statistics/statistics';
import { Models } from './components/models-page/models/models';

export const routes: Routes = [
  {
    path: 'home',
    component: Home,
    pathMatch: 'full',
  },
  {
    path: 'settings',
    component: SettingsComponent,
    pathMatch: 'full',
  },
  {
    path: 'datasets',
    component: DatasetsComponent,
    pathMatch: 'full',
  },
  {
    path: 'models',
    component: Models,
    pathMatch: 'full',
  },
  {
    path: 'statistics',
    component: Statistics,
    pathMatch: 'full',
  },
  {
    path: 'generator',
    component: Generator,
    pathMatch: 'full',
  },
  {
    path: '**',
    redirectTo: 'home',
    pathMatch: 'full',
  },
];
