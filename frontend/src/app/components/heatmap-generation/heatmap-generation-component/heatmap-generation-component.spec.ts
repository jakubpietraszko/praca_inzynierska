import { ComponentFixture, TestBed } from '@angular/core/testing';

import { HeatmapGenerationComponent } from './heatmap-generation-component';

describe('HeatmapGenerationComponent', () => {
  let component: HeatmapGenerationComponent;
  let fixture: ComponentFixture<HeatmapGenerationComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [HeatmapGenerationComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(HeatmapGenerationComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
