import { ComponentFixture, TestBed } from '@angular/core/testing';

import { HeatmapsSelector } from './heatmaps-selector';

describe('HeatmapsSelector', () => {
  let component: HeatmapsSelector;
  let fixture: ComponentFixture<HeatmapsSelector>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [HeatmapsSelector]
    })
    .compileComponents();

    fixture = TestBed.createComponent(HeatmapsSelector);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
