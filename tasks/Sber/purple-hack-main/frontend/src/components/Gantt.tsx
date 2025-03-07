import { Col, Row, Statistic, Typography } from 'antd';
import { Gantt, Task, ViewMode } from 'gantt-task-react';
import { observer } from 'mobx-react-lite';
import { ProjectInfo } from '../api/models';
import { useStores } from '../hooks/useStores';

type Props = {
    tasks: Task[];
    title: string;
    projectInfo: ProjectInfo;
};

const Diagram = observer(({ tasks, title, projectInfo }: Props) => {
    const { rootStore } = useStores();

    return (
        <>
            <Row gutter={16}>
                <Col span={8}>
                    <Statistic title='Стоимость' value={rootStore.getProjectCost(projectInfo)} />
                </Col>

                <Col span={8}>
                    <Statistic title='Длительность' value={projectInfo.tasks.rows[0].duration} />
                </Col>

                <Col span={8}>
                    <Statistic
                        title='Количество исполнителей'
                        value={rootStore.getProjectResourcesCount(projectInfo)}
                    />
                </Col>
            </Row>

            <Row style={{ marginTop: 20 }}>
                <Typography.Title level={2}>{title}</Typography.Title>
            </Row>

            <Gantt tasks={tasks} viewMode={ViewMode.Week} />
        </>
    );
});

export default Diagram;
