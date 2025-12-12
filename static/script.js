const featureColumns = [
    'Marital Status', 'Application mode', 'Application order', 'Course',
    'Daytime/evening attendance', 'Previous qualification', 'Previous qualification (grade)',
    'Nacionality', "Mother's qualification", "Father's qualification",
    "Mother's occupation", "Father's occupation", 'Admission grade', 'Displaced',
    'Educational special needs', 'Debtor', 'Tuition fees up to date', 'Gender',
    'Scholarship holder', 'Age at enrollment', 'International',
    'Curricular units 1st sem (credited)', 'Curricular units 1st sem (enrolled)',
    'Curricular units 1st sem (evaluations)', 'Curricular units 1st sem (approved)',
    'Curricular units 1st sem (grade)', 'Curricular units 1st sem (without evaluations)',
    'Curricular units 2nd sem (credited)', 'Curricular units 2nd sem (enrolled)',
    'Curricular units 2nd sem (evaluations)', 'Curricular units 2nd sem (approved)',
    'Curricular units 2nd sem (grade)', 'Curricular units 2nd sem (without evaluations)',
    'Unemployment rate', 'Inflation rate', 'GDP'
];

const exampleDropout = {
    'Marital Status': 1, 'Application mode': 17, 'Application order': 5, 'Course': 171,
    'Daytime/evening attendance': 1, 'Previous qualification': 1, 'Previous qualification (grade)': 122,
    'Nacionality': 1, "Mother's qualification": 19, "Father's qualification": 12,
    "Mother's occupation": 10, "Father's occupation": 5, 'Admission grade': 120,
    'Displaced': 0, 'Educational special needs': 0, 'Debtor': 0, 'Tuition fees up to date': 1,
    'Gender': 1, 'Scholarship holder': 0, 'Age at enrollment': 18, 'International': 0,
    'Curricular units 1st sem (credited)': 0, 'Curricular units 1st sem (enrolled)': 0,
    'Curricular units 1st sem (evaluations)': 0, 'Curricular units 1st sem (approved)': 0,
    'Curricular units 1st sem (grade)': 0, 'Curricular units 1st sem (without evaluations)': 0,
    'Curricular units 2nd sem (credited)': 0, 'Curricular units 2nd sem (enrolled)': 0,
    'Curricular units 2nd sem (evaluations)': 0, 'Curricular units 2nd sem (approved)': 0,
    'Curricular units 2nd sem (grade)': 0, 'Curricular units 2nd sem (without evaluations)': 0,
    'Unemployment rate': 10.8, 'Inflation rate': 1.4, 'GDP': 1.74
};

const exampleGraduate = {
    'Marital Status': 1, 'Application mode': 15, 'Application order': 1, 'Course': 9254,
    'Daytime/evening attendance': 1, 'Previous qualification': 1, 'Previous qualification (grade)': 160,
    'Nacionality': 1, "Mother's qualification": 1, "Father's qualification": 3,
    "Mother's occupation": 5, "Father's occupation": 8, 'Admission grade': 158,
    'Displaced': 0, 'Educational special needs': 0, 'Debtor': 0, 'Tuition fees up to date': 1,
    'Gender': 0, 'Scholarship holder': 1, 'Age at enrollment': 18, 'International': 0,
    'Curricular units 1st sem (credited)': 6, 'Curricular units 1st sem (enrolled)': 6,
    'Curricular units 1st sem (evaluations)': 6, 'Curricular units 1st sem (approved)': 6,
    'Curricular units 1st sem (grade)': 13.7, 'Curricular units 1st sem (without evaluations)': 0,
    'Curricular units 2nd sem (credited)': 6, 'Curricular units 2nd sem (enrolled)': 6,
    'Curricular units 2nd sem (evaluations)': 6, 'Curricular units 2nd sem (approved)': 6,
    'Curricular units 2nd sem (grade)': 13.7, 'Curricular units 2nd sem (without evaluations)': 0,
    'Unemployment rate': 13.9, 'Inflation rate': -0.3, 'GDP': 0.79
};

let probabilityChart = null;

function fillFormWithData(data) {
    featureColumns.forEach(col => {
        const input = document.getElementById(col);
        if (input && data[col] !== undefined) {
            input.value = data[col];
        }
    });
}

function fillExampleDropout() {
    fillFormWithData(exampleDropout);
    scrollToForm();
}


function fillExampleGraduate() {
    fillFormWithData(exampleGraduate);
    scrollToForm();
}

function scrollToForm() {
    document.querySelector('.section-input').scrollIntoView({ behavior: 'smooth' });
}

function resetForm() {
    document.getElementById('predictionForm').reset();
    document.getElementById('resultSection').style.display = 'none';
}

function getFormData() {
    const formData = {};
    featureColumns.forEach(col => {
        const input = document.getElementById(col);
        if (input) {
            const value = input.value.trim();
            if (value === '') {
                throw new Error(`Field "${col}" is required`);
            }
            formData[col] = parseFloat(value);
            if (isNaN(formData[col])) {
                throw new Error(`Field "${col}" must be a valid number`);
            }
        }
    });
    return formData;
}

async function submitPrediction() {
    const btn = event.target;
    const originalText = btn.textContent;
    
    try {
        const formData = getFormData();
        
        btn.disabled = true;
        btn.innerHTML = '<span class="loading"></span> Processing...';
        
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(formData)
        });
        
        const result = await response.json();
        
        if (!response.ok || !result.success) {
            showError(result.error || 'Prediction failed');
            return;
        }
        
        displayResults(result);
        scrollToResults();
        
    } catch (error) {
        showError(error.message);
    } finally {
        btn.disabled = false;
        btn.textContent = originalText;
    }
}

function displayResults(result) {
    const resultSection = document.getElementById('resultSection');
    const predictionBadge = document.getElementById('predictionBadge');
    const probabilityList = document.getElementById('probabilityList');
    const alertContainer = document.getElementById('alertContainer');

    if (alertContainer) {
        alertContainer.innerHTML = '';
    }
    
    const prediction = result.prediction.toLowerCase();
    predictionBadge.textContent = result.prediction;
    predictionBadge.className = `prediction-badge ${prediction}`;
    
    probabilityList.innerHTML = '';
    Object.entries(result.probabilities).forEach(([label, prob]) => {
        const item = document.createElement('div');
        item.className = 'prob-item';
        item.innerHTML = `
            <span class="prob-item-label">${label}</span>
            <span class="prob-item-value">${prob.toFixed(2)}%</span>
        `;
        probabilityList.appendChild(item);
    });
    
    updateChart(result.probabilities);
    
    resultSection.style.display = 'block';
}

function updateChart(probabilities) {
    const ctx = document.getElementById('probabilityChart');
    
    if (probabilityChart) {
        probabilityChart.destroy();
    }
    
    const labels = Object.keys(probabilities);
    const data = Object.values(probabilities);
    const colors = {
        'Dropout': 'rgba(239, 68, 68, 0.8)',
        'Enrolled': 'rgba(245, 158, 11, 0.8)',
        'Graduate': 'rgba(16, 185, 129, 0.8)'
    };
    const borderColors = {
        'Dropout': 'rgb(239, 68, 68)',
        'Enrolled': 'rgb(245, 158, 11)',
        'Graduate': 'rgb(16, 185, 129)'
    };
    
    probabilityChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Probability (%)',
                data: data,
                backgroundColor: labels.map(l => colors[l]),
                borderColor: labels.map(l => borderColors[l]),
                borderWidth: 2,
                borderRadius: 8,
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            indexAxis: 'y',
            plugins: {
                legend: {
                    display: false
                }
            },
            scales: {
                x: {
                    beginAtZero: true,
                    max: 100,
                    ticks: {
                        callback: function(value) {
                            return value + '%';
                        }
                    }
                }
            }
        }
    });
}

function showError(message) {
    const resultSection = document.getElementById('resultSection');
    const alertContainer = document.getElementById('alertContainer');

    if (alertContainer) {
        alertContainer.innerHTML = `
            <div class="alert alert-error">
                <strong>Error:</strong> ${message}
            </div>
        `;
    }
    resultSection.style.display = 'block';
    scrollToResults();
}

function scrollToResults() {
    document.getElementById('resultSection').scrollIntoView({ behavior: 'smooth' });
}

document.addEventListener('DOMContentLoaded', function() {
    const navToggle = document.getElementById('navToggle');
    const mainNav = document.getElementById('mainNav');
    const header = document.querySelector('.header');

    if (navToggle && mainNav) {
        navToggle.addEventListener('click', () => {
            mainNav.classList.toggle('is-active');
            header.classList.toggle('nav-open');
        });

        // Close nav when a link is clicked
        mainNav.addEventListener('click', (e) => {
            if (e.target.tagName === 'A') {
                mainNav.classList.remove('is-active');
                header.classList.remove('nav-open');
            }
        });
    }

    const form = document.getElementById('predictionForm');
    if (form) {
        form.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                e.preventDefault();
                submitPrediction();
            }
        });
    }
});
