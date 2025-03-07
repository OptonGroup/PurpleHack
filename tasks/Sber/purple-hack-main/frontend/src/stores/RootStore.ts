import { RcFile } from 'antd/es/upload';
import { makeAutoObservable } from 'mobx';
import {
    Assignment,
    Dependency,
    ProjectInfo,
    ProjectTask,
    UploadProjectFileParams,
} from '../api/models/Project';
import { Task } from 'gantt-task-react';
import { message } from 'antd';
import { TasksApiServiceInstanse } from '../api/TasksApiService';

export class RootStore {
    loading: boolean = false;
    file: RcFile | null = null;
    uploadedProjectInfo: ProjectInfo | null = null;
    calculatedProjectInfo: ProjectInfo | null = null;
    uploadedGanttTasks: Task[] | null = null;
    calculatedGanttTasks: Task[] | null = null;

    constructor() {
        makeAutoObservable(this);
    }

    uploadFile(file: RcFile) {
        if (file) {
            const reader = new FileReader();

            reader.onload = () => {
                try {
                    const parsedData = JSON.parse(reader.result as string);
                    this.uploadedProjectInfo = parsedData;
                    this.file = file;

                    if (this.uploadedProjectInfo) {
                        this.uploadedGanttTasks = this.mapToTasksPipeline(this.uploadedProjectInfo);
                    }
                } catch (error) {
                    console.error('Error parsing JSON:', error);
                }
            };

            reader.readAsText(file);
        }
    }

    mapProjectInfoToTasks(tasks: ProjectTask[]): Task[] {
        const mappedTasks: Task[] = [];

        tasks.forEach((task) => {
            if (task.children && task.children.length > 0) {
                const mappedChildren = this.mapProjectInfoToTasks(task.children);

                mappedTasks.push({
                    id: task.id,
                    type: 'project',
                    name: task.name + ` id: ${task.id}`,
                    start: new Date(task.startDate),
                    end: new Date(task.endDate),
                    progress: task.percentDone,
                    isDisabled: task.inactive,
                    project: task.parentId,
                    dependencies: [],
                    hideChildren: false,
                });

                mappedTasks.push(...mappedChildren);
            } else {
                mappedTasks.push({
                    id: task.id,
                    type: 'task',
                    name: task.name + `; id: ${task.id}`,
                    start: new Date(task.startDate),
                    end: new Date(task.endDate),
                    progress: task.percentDone,
                    isDisabled: task.inactive,
                    project: task.parentId,
                    dependencies: [],
                    hideChildren: false,
                });
            }
        });

        return mappedTasks;
    }

    addDependenciesToTasks(tasks: Task[], dependencies: Dependency[]): Task[] {
        tasks.forEach((task) => {
            const taskDependencies = dependencies.filter((dependency) => dependency.to === task.id);
            taskDependencies.forEach((dependency) => {
                const dependentTask = tasks.find((t) => t.id === dependency.from);
                if (dependentTask) {
                    task.dependencies?.push(dependentTask.id);
                }
            });
        });

        return tasks;
    }

    addResoursesToTasks(tasks: Task[], assignments: Assignment[]): Task[] {
        tasks.forEach((task) => {
            const taskAssigments = assignments.filter((assignment) => assignment.event === task.id);

            taskAssigments.forEach((assigment) => {
                const task = tasks.find((t) => t.id === assigment.event);

                if (task) {
                    task.name += ` (${assigment.resource.slice(-4)})`;
                }
            });
        });

        return tasks;
    }

    async calculate({ duration, file, price, resource }: UploadProjectFileParams) {
        this.loading = true;

        console.log(this.file);

        TasksApiServiceInstanse.calculate({ file, duration, price, resource })
            .then((data) => {
                this.calculatedProjectInfo = data;

                this.calculatedGanttTasks = this.mapToTasksPipeline(data);

                message.success('Файл успешно обработан');
            })
            .catch(() => {
                message.error('Ошибка обработки файла. Попробуйте еще раз.');
            })
            .finally(() => {
                this.loading = false;
            });
    }

    private mapToTasksPipeline(projectInfo: ProjectInfo): Task[] {
        return this.addResoursesToTasks(
            this.addDependenciesToTasks(
                this.mapProjectInfoToTasks(projectInfo.tasks.rows),
                projectInfo.dependencies.rows as Dependency[]
            ),
            projectInfo.assignments.rows as Assignment[]
        );
    }

    getProjectCost(projectInfo: ProjectInfo): number {
        let totalCost = 0;

        const tasks = this.getAllLeafTasksRecursively(projectInfo.tasks.rows);

        const taskToResourceMap = this.getTaskToResourcesMap(projectInfo);

        tasks.forEach((task) => {
            const resource = projectInfo.resources.rows.find(
                (resource) => resource.id === taskToResourceMap.get(task.id)
            );

            if (
                resource &&
                resource.name &&
                typeof resource.name === 'string' &&
                task &&
                task.effort &&
                typeof task.effort === 'number'
            ) {
                try {
                    totalCost += task.effort * +resource.name.match(/(?<=\()\d+/g)![0];
                } catch (error) {
                    console.log(error);
                }
            }
        });

        return totalCost;
    }

    getProjectResourcesCount(projectInfo: ProjectInfo): number {
        const resources = new Set<string>();

        projectInfo.assignments.rows.forEach((assignment) => {
            if (assignment.resource) resources.add(assignment.resource);
        });

        return resources.size;
    }

    private getTaskToResourcesMap(projectInfo: ProjectInfo): Map<string, string> {
        const taskToResourceMap = new Map<string, string>();

        projectInfo.assignments.rows.forEach((assignment) => {
            taskToResourceMap.set(assignment.event, assignment.resource);
        });

        return taskToResourceMap;
    }

    private getAllLeafTasksRecursively(tasks: ProjectTask[]): ProjectTask[] {
        const leafTasks: ProjectTask[] = [];

        tasks.forEach((task) => {
            if (task.children && task.children.length > 0) {
                leafTasks.push(...this.getAllLeafTasksRecursively(task.children));
            } else {
                leafTasks.push(task);
            }
        });

        return leafTasks;
    }

    downloadFile() {
        if (this.calculatedProjectInfo) {
            const data = JSON.stringify(this.calculatedProjectInfo);

            const blob = new Blob([data], { type: 'application/json' });
            const url = URL.createObjectURL(blob);

            const a = document.createElement('a');
            a.href = url;
            a.download = 'result.json';
            document.body.appendChild(a);
            a.click();
            a.remove();
        }
    }
}
