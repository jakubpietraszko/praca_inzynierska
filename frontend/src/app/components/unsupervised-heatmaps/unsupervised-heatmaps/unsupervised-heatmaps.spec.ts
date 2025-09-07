import { ComponentFixture, TestBed } from '@angular/core/testing';

import { UnsupervisedHeatmaps } from './unsupervised-heatmaps';

describe('UnsupervisedHeatmaps', () => {
  let component: UnsupervisedHeatmaps;
  let fixture: ComponentFixture<UnsupervisedHeatmaps>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [UnsupervisedHeatmaps]
    })
    .compileComponents();

    fixture = TestBed.createComponent(UnsupervisedHeatmaps);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
