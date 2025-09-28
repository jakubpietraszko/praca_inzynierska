import { Routes } from '@angular/router';
import { SettingsComponent } from './components/settings/settings-component/settings-component';
import { Generator } from './components/generator-page/generator';
import { Home } from './components/home-page/home/home';
import { Statistics } from './components/statistics-page/statistics';
import { Models } from './components/models-page/models/models';
import { Datasets } from './components/datasets-page/datasets';

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
    component: Datasets,
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
