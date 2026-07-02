// Chart.js 인스턴스 저장용
let trendChart;

// 1. 초기 트렌드 차트 생성 (Line Chart)
function initTrendChart() {
    const ctxTrend = document.getElementById('trendChart').getContext('2d');
    trendChart = new Chart(ctxTrend, {
        type: 'line',
        data: {
            labels: ['24.01', '24.06', '24.12', '25.06', '25.12', '26.05'],
            datasets: [{
                label: '위반 지수 (위험도)',
                data: [12, 19, 15, 25, 32, 45],
                borderColor: '#ff4b4b',
                backgroundColor: 'rgba(255, 75, 75, 0.2)',
                fill: true,
                tension: 0.4
            }, {
                label: '소비자물가지수(CPI)',
                data: [105, 108, 112, 118, 122, 128],
                borderColor: '#00d4ff',
                borderDash: [5, 5],
                fill: false,
                tension: 0.1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { labels: { color: '#aaa' } }
            },
            scales: {
                y: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#888' } },
                x: { grid: { display: false }, ticks: { color: '#888' } }
            }
        }
    });
}

// 2. 지역별 위험 레이더 차트 (Radar Chart)
function initRadarChart() {
    const ctxRadar = document.getElementById('radarChart').getContext('2d');
    new Chart(ctxRadar, {
        type: 'radar',
        data: {
            labels: ['서울', '경기', '부산', '대구', '대전'],
            datasets: [{
                label: '현재 위험 지수',
                data: [95, 88, 72, 65, 80],
                backgroundColor: 'rgba(0, 212, 255, 0.2)',
                borderColor: '#00d4ff',
                pointBackgroundColor: '#00d4ff'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            },
            scales: {
                r: {
                    angleLines: { color: 'rgba(255,255,255,0.1)' },
                    grid: { color: 'rgba(255,255,255,0.1)' },
                    pointLabels: { color: '#aaa', font: { size: 12 } },
                    ticks: { display: false }
                }
            }
        }
    });
}

// 3. 시뮬레이션 인터랙션 (Simulation Logic)
const cpiSlider = document.getElementById('cpiSlider');
const thiSlider = document.getElementById('thiSlider');
const cpiValDisp = document.getElementById('cpiVal');
const riskLevelDisp = document.getElementById('riskLevel');
const alertLog = document.getElementById('alertLog');

function addLog(message) {
    const time = new Date().toLocaleTimeString();
    const div = document.createElement('div');
    div.className = 'log-item';
    div.innerText = `[${time}] ${message}`;
    alertLog.prepend(div);
}

function updateSimulation() {
    const cpi = parseFloat(cpiSlider.value);
    const thi = parseFloat(thiSlider.value);
    cpiValDisp.innerText = cpi;

    // 차트 데이터 업데이트 (시뮬레이션 효과)
    // 물가가 오를수록 마지막 데이터 포인트가 상승하도록 시뮬레이션
    const baseData = [12, 19, 15, 25, 32, 45];
    const newData = baseData.map((v, i) => {
        if (i > 3) return v + (cpi * 2.5) + (thi * 3);
        return v;
    });
    trendChart.data.datasets[0].data = newData;
    trendChart.update();

    // 위험도 등급 판정
    const totalRisk = cpi * 1.5 + thi * 2;
    if (totalRisk > 10) {
        riskLevelDisp.innerText = "🚨 위험 (CRITICAL)";
        riskLevelDisp.className = "danger";
        addLog(`경보: 물가/기후 복합 압박으로 인한 위생 위험지수 임계점 돌파!`);
    } else if (totalRisk > 4) {
        riskLevelDisp.innerText = "⚠️ 주의";
        riskLevelDisp.className = "warning";
        addLog(`분석: 특정 영세 상권의 위생 타협 징후 포착`);
    } else {
        riskLevelDisp.innerText = "✅ 안전";
        riskLevelDisp.className = "safe";
    }
}

// 4. 내비게이션 인터랙션
const navItems = document.querySelectorAll('.nav-item');
navItems.forEach(item => {
    item.addEventListener('click', () => {
        navItems.forEach(n => n.classList.remove('active'));
        item.classList.add('active');
        addLog(`${item.innerText} 화면으로 전환합니다.`);
    });
});

// 초기화
document.addEventListener('DOMContentLoaded', () => {
    initTrendChart();
    initRadarChart();
    
    cpiSlider.addEventListener('input', updateSimulation);
    thiSlider.addEventListener('input', updateSimulation);
    
    addLog("시스템 정상 가동 중. 실시간 데이터 스트리밍 활성화.");
});
