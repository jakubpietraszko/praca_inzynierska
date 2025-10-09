import { ChangeDetectionStrategy, Component } from '@angular/core';

@Component({
  selector: 'app-content-different-statistics',
  imports: [],
  templateUrl: './content-different-statistics.html',
  styleUrl: './content-different-statistics.css',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class ContentDifferentStatistics {}
