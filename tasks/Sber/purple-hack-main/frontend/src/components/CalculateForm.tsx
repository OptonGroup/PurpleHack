import { Button, Col, Form, InputNumber, Row } from 'antd';
import { observer } from 'mobx-react-lite';
import { useStores } from '../hooks/useStores';

const CalculateForm = observer(() => {
    const { rootStore } = useStores();
    const [form] = Form.useForm();

    const onFinish = ({
        duration,
        price,
        resource,
    }: {
        duration: number | undefined;
        price: number | undefined;
        resource: number | undefined;
    }) => {
        if (rootStore.file) {
            rootStore.calculate({
                file: rootStore.file,
                duration: duration ?? 0,
                price: price ?? 0,
                resource: resource ?? 0,
            });
        }
    };

    return (
        <Form
            layout='vertical'
            name='basic'
            initialValues={{ remember: true }}
            onFinish={onFinish}
            autoComplete='off'
            form={form}
            style={{ width: '100%' }}
        >
            <Row style={{ width: '100%' }} gutter={[16, 16]}>
                <Col span={8}>
                    <Button
                        block
                        type={'primary'}
                        style={{ whiteSpace: 'normal', height: 'auto', marginBottom: '10px' }}
                        onClick={() => {
                            form.setFieldsValue({ duration: 1, price: 0, resource: 0 });
                        }}
                    >
                        Оптимизировать <br /> по длительности
                    </Button>

                    <Form.Item style={{ width: '100%' }} label='Вес "длительности"' name='duration'>
                        <InputNumber precision={2} style={{ width: '100%' }} />
                    </Form.Item>
                </Col>

                <Col span={8}>
                    <Button
                        block
                        type={'primary'}
                        style={{ whiteSpace: 'normal', height: 'auto', marginBottom: '10px' }}
                        onClick={() => {
                            form.setFieldsValue({ duration: 0, price: 1, resource: 0 });
                        }}
                    >
                        Оптимизировать <br /> по стоимости
                    </Button>

                    <Form.Item label='Вес "стоимости"' name='price'>
                        <InputNumber precision={2} style={{ width: '100%' }} />
                    </Form.Item>
                </Col>

                <Col span={8}>
                    <Button
                        block
                        type={'primary'}
                        style={{ whiteSpace: 'normal', height: 'auto', marginBottom: '10px' }}
                        onClick={() => {
                            form.setFieldsValue({ duration: 0, price: 0, resource: 1 });
                        }}
                    >
                        Оптимизировать <br /> по ресурсам
                    </Button>

                    <Form.Item label='Вес "ресурсов"' name='resource'>
                        <InputNumber precision={2} style={{ width: '100%' }} />
                    </Form.Item>
                </Col>
            </Row>

            <Form.Item>
                <Button loading={rootStore.loading} size='large' type='primary' htmlType='submit'>
                    Расчитать
                </Button>
            </Form.Item>
        </Form>
    );
});

export default CalculateForm;
