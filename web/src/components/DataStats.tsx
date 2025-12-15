import React from 'react';

interface DataStatsType {
  papers_count: number;
  journals_count: number;
  top_concepts: Array<{
    name: string;
    count: number;
  }>;
}

interface DataStatsProps {
  dataStats: DataStatsType | null;
}

const DataStats: React.FC<DataStatsProps> = ({ dataStats }) => {
  if (!dataStats) {
    return (
      <div className="dashboard-card">
        <h3 className="card-title">📊 Статистика данных</h3>
        <div className="loading">Загрузка...</div>
      </div>
    );
  }

  const totalConcepts = dataStats.top_concepts?.reduce((sum, concept) => sum + concept.count, 0) || 0;

  return (
    <div className="dashboard-card">
      <h3 className="card-title">📊 Статистика данных</h3>

      {/* Main Stats */}
      <div style={{ marginBottom: '20px' }}>
        <div style={{
          display: 'grid',
          gridTemplateColumns: '1fr 1fr',
          gap: '15px',
          marginBottom: '15px'
        }}>
          <div style={{ textAlign: 'center' }}>
            <div className="metric-value" style={{ color: '#3b82f6' }}>
              {(dataStats.papers_count || 0).toLocaleString()}
            </div>
            <div className="metric-label">📄 Статей в базе</div>
          </div>

          <div style={{ textAlign: 'center' }}>
            <div className="metric-value" style={{ color: '#8b5cf6' }}>
              {(dataStats.journals_count || 0).toLocaleString()}
            </div>
            <div className="metric-label">📚 Журналов</div>
          </div>
        </div>

        <div style={{ textAlign: 'center' }}>
          <div className="metric-value" style={{ fontSize: '1.5em', color: '#f59e0b' }}>
            {totalConcepts.toLocaleString()}
          </div>
          <div className="metric-label">🏷️ Всего концептов</div>
        </div>
      </div>

      {/* Top Concepts */}
      {dataStats.top_concepts && dataStats.top_concepts.length > 0 && (
        <div>
          <h4 style={{
            marginBottom: '15px',
            fontSize: '1em',
            color: '#374151',
            borderBottom: '1px solid #e5e7eb',
            paddingBottom: '5px'
          }}>
            🔥 Топ концепты
          </h4>

          <div style={{ maxHeight: '300px', overflowY: 'auto' }}>
            {dataStats.top_concepts.map((concept, index) => {
              const percentage = totalConcepts > 0 ? (concept.count / totalConcepts * 100) : 0;

              return (
                <div key={index} style={{
                  marginBottom: '12px',
                  padding: '8px',
                  border: '1px solid #e5e7eb',
                  borderRadius: '6px',
                  background: index < 3 ? '#fef3c7' : '#f8fafc'
                }}>
                  <div style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                    marginBottom: '5px'
                  }}>
                    <div style={{
                      fontWeight: 'bold',
                      color: '#374151',
                      fontSize: '0.9em'
                    }}>
                      {index < 3 && ['🥇', '🥈', '🥉'][index]} {concept.name}
                    </div>
                    <div style={{
                      fontSize: '0.9em',
                      fontWeight: 'bold',
                      color: '#667eea'
                    }}>
                      {concept.count}
                    </div>
                  </div>

                  {/* Progress bar */}
                  <div className="progress-bar" style={{ height: '6px' }}>
                    <div
                      className="progress-fill"
                      style={{
                        width: `${Math.min(percentage, 100)}%`,
                        background: index < 3 ?
                          ['linear-gradient(90deg, #fbbf24, #f59e0b)',
                           'linear-gradient(90deg, #d1d5db, #9ca3af)',
                           'linear-gradient(90deg, #cd7c2f, #92400e)'][index] :
                          'linear-gradient(90deg, #667eea, #764ba2)'
                      }}
                    />
                  </div>

                  <div style={{
                    fontSize: '0.8em',
                    color: '#666',
                    marginTop: '3px'
                  }}>
                    {percentage.toFixed(1)}% от всех концептов
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Data Quality Indicators */}
      <div style={{
        marginTop: '20px',
        padding: '10px',
        background: '#f0f9ff',
        border: '1px solid #bae6fd',
        borderRadius: '8px'
      }}>
        <div style={{ fontWeight: 'bold', marginBottom: '5px', color: '#0369a1' }}>
          📈 Качество данных:
        </div>

        <div style={{ fontSize: '0.8em', color: '#0369a1' }}>
          <div>• Уникальных концептов: {dataStats.top_concepts?.length || 0}</div>
          <div>• Средняя плотность: {totalConcepts > 0 && dataStats.papers_count ? (totalConcepts / dataStats.papers_count).toFixed(1) : 0} концептов/статья</div>
          <div>• Покрытие данных: {(dataStats.papers_count || 0) > 100 ? 'Отличное' : (dataStats.papers_count || 0) > 50 ? 'Хорошее' : 'Требует расширения'}</div>
        </div>
      </div>

      {/* Recommendations */}
      {(dataStats.papers_count || 0) < 100 && (
        <div style={{
          marginTop: '15px',
          padding: '10px',
          background: '#fef3c7',
          border: '1px solid #fcd34d',
          borderRadius: '8px'
        }}>
          <div style={{ fontWeight: 'bold', marginBottom: '5px', color: '#92400e' }}>
            💡 Рекомендация:
          </div>
          <div style={{ fontSize: '0.8em', color: '#92400e' }}>
            Для лучшего качества обучения рекомендуется иметь минимум 100-200 статей.
            Загрузите больше данных через панель управления.
          </div>
        </div>
      )}
    </div>
  );
};

export default DataStats;
